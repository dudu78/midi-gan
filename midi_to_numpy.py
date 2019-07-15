from itertools import chain

import mido
import numpy as np

SAMPLES_PER_MEASURE = 2 * 96
MEASURES_IN_OUTPUT = 16
TOTAL_OUTPUT_TIME_LENGTH = SAMPLES_PER_MEASURE * MEASURES_IN_OUTPUT


def detect_time_signature(mid):
    """
    returns the ticks per measure of the track, raising an exception if multiple.
    (note: 1 measure = 4 beats = 4 quarter notes)
    :param mid: mide.MidiFile object as input
    :return: the ticks per measure of the track
    """
    has_time_sig = False
    ticks_per_beat = mid.ticks_per_beat
    ticks_per_measure = 4 * ticks_per_beat
    flag_warning = False
    for i, track in enumerate(mid.tracks):
        for msg in track:
            if msg.type == 'time_signature':
                new_tpm = msg.numerator * ticks_per_beat * 4 / msg.denominator
                if has_time_sig and new_tpm != ticks_per_measure:
                    flag_warning = True
                ticks_per_measure = new_tpm
                has_time_sig = True
    if flag_warning:
        raise ValueError('multiple distinct time signatures')
    return ticks_per_measure


def numpy_from_midi(midi):
    """
    processes a midi to ChannelArrays
    :param midi: mido.MidiFile object as input
    :return: iterator over ChannelArrays, extracted from the midi
    """
    ticks_per_measure = detect_time_signature(midi)
    note_start_times = {}
    note_start_velocities = {}
    channels = {}
    finished_channels = []
    absolute_time = 0
    samples_per_tick = SAMPLES_PER_MEASURE / ticks_per_measure

    def curr_array_time():
        """
        :return: time in the array that corresponds to current time in the midi
        """
        return int(absolute_time * samples_per_tick)
        # return absolute_time // (midi.ticks_per_beat // 8)

    def finish_note(note, stop_velocity, channel_num):
        """
        :param note: note number to finish
        :param stop_velocity: stopping velocity for the note (starting velocity is saved in note_start_velocities array)
        :param channel_num: channel number of the note to finish
        """
        start_velocity = note_start_velocities[channel_num][note]
        start_time = note_start_times[channel_num][note]
        stop_time = curr_array_time()
        assert start_velocity <= 1 and stop_velocity <= 1
        if stop_time - start_time == 0:
            stop_time += 1
        on_strip = np.linspace(
            start_velocity,
            stop_velocity,
            num=stop_time - start_time
        )
        channels[channel_num].array[note][start_time:stop_time] = on_strip
        note_start_times[channel_num][note] = -1  # mark note as finished

    for track in midi.tracks:
        print('track:', track)
        for msg in track:
            # print('', msg)
            if msg.type == 'control_change':
                if msg.control == 0:  # bank change, start of a new track
                    absolute_time = 0
                    continue

            absolute_time += msg.time
            if msg.type == 'program_change':
                print(msg)
                if msg.channel in channels:  # program change for existing channel
                    finished_channels.append(channels[msg.channel])  # archive channel
                channels[msg.channel] = ChannelArray(msg.channel, program=msg.program)
                note_start_times[msg.channel] = np.full((128,), -1)
                note_start_velocities[msg.channel] = np.full((128,), 0.0)
            elif msg.type == 'note_on':
                if msg.channel != 1: continue  # FIXME
                if curr_array_time() >= TOTAL_OUTPUT_TIME_LENGTH:
                    continue
                if msg.velocity == 0:  # same as note_off
                    # ignore
                    print('ignoring velocity 0')
                    continue

                if note_start_times[msg.channel][msg.note] != -1:  # note already started
                    print('unfinished note %d at time %d starts again, finishing'
                          % (msg.note, curr_array_time()))
                    finish_note(msg.note, msg.velocity / (127 * 2), msg.channel)

                note_start_velocities[msg.channel][msg.note] = msg.velocity / 127
                note_start_times[msg.channel][msg.note] = curr_array_time()

            elif msg.type == 'note_off':
                if msg.channel != 1: continue  # FIXME
                if curr_array_time() >= TOTAL_OUTPUT_TIME_LENGTH:
                    continue
                if note_start_times[msg.channel][msg.note] == -1:
                    print('note %d at time %d to finish is already finished, finishing'
                          % (msg.note, curr_array_time()))
                    continue
                finish_note(msg.note, msg.velocity / 127, msg.channel)
    return chain(finished_channels, channels.values())


def shift_left_array(array):
    """
    >>> shift_left_array([5,4,9,2])
    [0, 5, 4, 9 ]
    """
    res = np.roll(array, -1)
    res[-1] = 0
    return res


def shift_right_array(array):
    """
    >>> shift_right_array([5,4,9,2])
    [0, 5, 4, 9]
    """
    res = np.roll(array, 1)
    res[0] = 0
    return res


def numpy_to_midi_track(chan_array, ticks_per_measure):
    """
    turns a channel to a midi track, complete with track start and track end messages
    :param chan_array: a ChannelArray input object
    :param ticks_per_measure: desired ticks per measure in the output, used to calculate ticks per sample
    :return: mido.MidiTrack full of messages
    """
    ticks_per_sample = ticks_per_measure / SAMPLES_PER_MEASURE
    chan_num = chan_array.channel_num
    array = chan_array.array
    shift_right = shift_right_array(array)
    deriv = array - shift_right
    note_starts = deriv > 0
    note_starts = (shift_right == 0) & (array != 0)
    note_stops = (shift_left_array(array) == 0) & (array != 0)
    array = array.T

    note_starts = note_starts.T
    note_stops = note_stops.T
    # now everything is represented time-wise instead of note-wise
    # array[time][note] == velocity
    did_note_start = np.full((128,), False)
    delta_time = 0.0
    track = mido.MidiTrack()
    # add new track messages
    track.extend([
        mido.MetaMessage('track_name', name=chan_array.name),
        mido.Message('control_change', control=0,
                     channel=chan_num),
        mido.Message('program_change', channel=chan_num,
                     program=chan_array.program)
    ])
    for vel_arr, is_starting_arr, is_stopping_arr in zip(array,
                                                         note_starts,
                                                         note_stops):
        for note_num, (vel, is_starting, is_stopping) in enumerate(zip(vel_arr,
                                                                       is_starting_arr,
                                                                       is_stopping_arr)):
            if is_starting:
                if did_note_start[note_num]:  # starting an already started note
                    # end current one
                    track.append(mido.Message('note_off', note=note_num,
                                              velocity=int(127 * vel),
                                              time=int(delta_time), channel=chan_num))
                    delta_time -= int(delta_time)
                track.append(mido.Message('note_on', note=note_num,
                                          velocity=int(127 * vel),
                                          time=int(delta_time), channel=chan_num))
                did_note_start[note_num] = True
                delta_time -= int(delta_time)
            if is_stopping:
                if did_note_start[note_num]:  # ending a started note
                    track.append(mido.Message('note_off', note=note_num,
                                              velocity=int(127 * vel),
                                              time=int(delta_time), channel=chan_num))
                    delta_time -= int(delta_time)
                    did_note_start[note_num] = False
                else:
                    print("cannot end an unstarted note")

        delta_time += ticks_per_sample  # next sample

    track.append(mido.MetaMessage('end_of_track'))
    return track


class ChannelArray:
    """
    represents a midi track as a numpy array, with some extra info that does not fit in the track e.g. program number.
    """

    def __init__(self, channel_num, program=None, name='', array=None):
        self.program = program
        self.name = name
        self.channel_num = channel_num
        self.array = array or np.zeros((128, TOTAL_OUTPUT_TIME_LENGTH))
        # self.array[note][time] is velocity

    def _arr_info(self):
        playing = (self.array > 0).T
        # a[time][note] == (is note playing at that time)
        start_ind = np.argmax(playing) // playing.shape[1]
        stop_ind = (len(playing.flatten()) - np.argmax(playing[::-1])) // playing.shape[1]
        return ('starting %4d stopping %4d'
                % (start_ind, stop_ind))

    def __repr__(self):
        return 'channelArray %s#%d with prog %3d. %s' \
               % (self.name + ' ' if self.name else '',
                  self.channel_num, self.program, self._arr_info())


def map_to_4_channels(channels):
    """
    maps multiple channels to only 4 channels, intended for reduction the number of channels for an AI to process
    :param channels: an iterable of ChannelArrays
    :return: an list of 4 ChannelArrays
    """
    # create the four channels:
    drums = ChannelArray(1, name='drums', program=118)  # TODO find a good drums program number
    piano = ChannelArray(2, name='piano', program=4)
    bass = ChannelArray(3, name='bass', program=32)
    guitar = ChannelArray(4, name='guitar', program=25)
    # map the given channels to our four
    for channel in channels:
        print('mapping %d to ' % channel.program, end='')
        if channel.program == 0 or channel.program >= 111:
            drums.array += channel.array
            print('drums')
        elif (24 <= channel.program <= 30 or
              40 <= channel.program <= 69 or
              104 <= channel.program <= 110):
            guitar.array += channel.array
            print('guitar')
        elif (31 <= channel.program <= 39 or
              81 <= channel.program <= 96):
            bass.array += channel.array
            print('bass')
        else:
            piano.array += channel.array
            print('piano')
    # some values may overflow above 1.0 resulting from the additions,
    # lets reset them back to 1.0
    for out_chan in [drums, piano, bass, guitar]:
        out_chan.array = np.minimum(out_chan.array, 1.0)
    return drums, piano, bass, guitar


tests = [
    'tests/commend.mid',
    'tests/dgate007.mid',
    'tests/doom1e1.mid',
    'tests/doom1e1_src_messages.txt',
    'tests/ff3veldt.mid',
    'tests/g3-intro.mid',
    'tests/ky1_24.mid',
    'tests/ky2-78.mid',
    'tests/ky2-83.mid',
    'tests/sick.mid',
    'tests/strike36.mid',
    'tests/strike41.mid',
    'tests/zelda1.mid',
]


def write_msgs_to_file(filename, midi):
    """writes messages of a midi to a file in readable form, for debugging"""
    f = open(filename, 'w')
    for track in midi.tracks:
        f.write(str(track) + '\n')
        for msg in track:
            f.write(' ' + str(msg) + '\n')


def there_and_back_again(filename):
    """
    converts a midi to numpy arrays and back, used for testing the conversion
    :param filename: an input file name of a midi file
    """
    print('testing on', filename)
    src_midi = mido.MidiFile(filename)
    ticks_per_measure = detect_time_signature(src_midi)
    print(src_midi)
    print(src_midi.ticks_per_beat)
    channels = numpy_from_midi(src_midi)
    # channels = map_to_4_channels(channels.values())
    midi = mido.MidiFile(ticks_per_beat=src_midi.ticks_per_beat)

    infos = []
    for chan_array in channels:
        infos.append(str(chan_array))
        back_track = numpy_to_midi_track(chan_array, ticks_per_measure)
        midi.tracks.append(back_track)
    infos.sort()
    print('\n'.join(infos))
    midi.save('result.mid')
    write_msgs_to_file(filename.split('.')[0] + '_result_messages.txt', midi)
    write_msgs_to_file(filename.split('.')[0] + '_src_messages.txt', src_midi)


input_filename = tests[0]
print('testing on file', input_filename)
there_and_back_again(input_filename)

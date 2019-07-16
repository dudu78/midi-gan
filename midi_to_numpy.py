from itertools import chain

import mido
import numpy as np

SAMPLES_PER_MEASURE = 48
MEASURES_IN_OUTPUT = 32
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

    def create_channel(channel_num, program):
        channels[channel_num] = ChannelArray(msg.channel, program=program)
        note_start_times[channel_num] = np.full((128,), -1)
        note_start_velocities[channel_num] = np.full((128,), 0.0)

    for track in midi.tracks:
        # print('track:', track)
        pass
        for msg in track:
            if msg.type == 'control_change':
                if msg.channel not in channels:  # create new channel
                    channels[msg.channel] = ChannelArray(msg.channel, program=None)
                    note_start_times[msg.channel] = np.full((128,), -1)
                    note_start_velocities[msg.channel] = np.full((128,), 0.0)
                if msg.control == 0:  # bank change, start of a new track
                    absolute_time = 0
                    continue
            elif msg.type == 'end_of_track':
                absolute_time = 0
                continue
            elif msg.type == 'marker':
                continue
            absolute_time += msg.time

            if msg.type == 'program_change':
                # print(msg)
                if msg.channel in channels:  # program change for existing channel
                    if channels[msg.channel].program is None:  # first time setting program, no need to create new channel
                        channels[msg.channel].program = msg.program
                    else:
                        # introduce this as a new channel
                        finished_channels.append(channels[msg.channel])  # archive channel
                        create_channel(msg.channel, msg.program)
                else:
                    # create new channel
                    create_channel(msg.channel, msg.program)
            elif msg.type == 'note_on':
                if curr_array_time() >= TOTAL_OUTPUT_TIME_LENGTH:
                    continue
                if msg.channel not in channels:
                    create_channel(msg.channel, None)
                if msg.velocity == 0:  # same as note_off
                    if note_start_times[msg.channel][msg.note] != -1:  # note started, lets finish
                        finish_note(msg.note, msg.velocity / 127, msg.channel)
                    continue

                if note_start_times[msg.channel][msg.note] != -1:  # note already started
                    print('unfinished note %d at time %d starts again, finishing'
                          % (msg.note, curr_array_time()))
                    finish_note(msg.note, msg.velocity / (127 * 2), msg.channel)

                note_start_velocities[msg.channel][msg.note] = msg.velocity / 127
                note_start_times[msg.channel][msg.note] = curr_array_time()

            elif msg.type == 'note_off':
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
    [4,9,2,5]
    """
    res = np.roll(array, -1)
    res[-1] = 0
    return res


def shift_right_array(array):
    """
    >>> shift_right_array([5,4,9,2])
    [0,5,4,9]
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
    program = chan_array.program or 0

    shift_right = shift_right_array(array)
    note_starts = (shift_right == 0) & (array != 0)
    note_stops = (shift_left_array(array) == 0) & (array != 0)
    note_stops = shift_right_array(note_stops)  # FIXME
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
                     program=program)
    ])
    if chan_array.messages:
        track.extend(chan_array.messages)

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

    def __init__(self, channel_num, program=None, name='',
                 array=None, messages=None):
        self.program = program
        self.name = name
        self.channel_num = channel_num
        self.array = array or np.zeros((128, TOTAL_OUTPUT_TIME_LENGTH))
        self.messages = messages
        # self.array[note][time] is velocity

    def _arr_info(self):
        """info on array start and end indexes, used for debugging"""
        playing = (self.array > 0).T
        if not np.any(playing):  # all is False
            return 'empty'
        start_ind = np.argmax(playing) // playing.shape[1]  # first True
        stop_ind = (len(playing.flatten()) - np.argmax(playing[::-1])) // playing.shape[1]  # last True
        return ('starting %4d stopping %4d'
                % (start_ind, stop_ind))

    def __repr__(self):
        return 'channelArray {}#{} with prog {:3d}. {}' \
            .format(self.name + ' ' if self.name else '',
                    self.channel_num,
                    self.program if self.program is not None else -1,
                    self._arr_info())


def map_to_4_channels(channels):
    """
    maps multiple channels to only 4 channels, intended for reduction the number of channels for an AI to process
    :param channels: an iterable of ChannelArrays
    :return: an list of 4 ChannelArrays
    """
    # create the four channels:
    drums = ChannelArray(9, name='drums', program=0)  # channel_num has to be 9
    # <meta message sequencer_specific data=(5, 15, 10, 1) time=0>

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
    try:
        src_midi = mido.MidiFile(filename)
    except EOFError:
        print('error with midi file, exiting')
        return
    ticks_per_measure = detect_time_signature(src_midi)
    print(src_midi)
    print(src_midi.ticks_per_beat)
    channels = numpy_from_midi(src_midi)
    channels = map_to_4_channels(channels)
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


tests = [
    'tests/CTOceanPalaceByCryogen.mid',
    'tests/FireEmblem4_Crisis1.mid',
    'tests/g3-intro.mid',
    'tests/gmbalrog.mid',
    'tests/MushiHS_Stage1.mid',
    'tests/rcrstatus.mid',
    'tests/sf2endch.mid',
    'tests/sick.mid',
    'tests/tmntovrw.mid',
    'tests/turrican2_mstars.mid',
    'tests/zelda1.mid',
]

input_filename = tests[2]
there_and_back_again(input_filename)

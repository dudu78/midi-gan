#!/usr/bin/env python

import mido
import os
import time
from multiprocessing import Pool
from midi_to_numpy import *
import numpy as np
from pathlib import Path
import tensorflow as tf
import random


AUTOTUNE = tf.data.experimental.AUTOTUNE
SONG_SHAPE = (12, 7, TOTAL_OUTPUT_TIME_LENGTH, 1)


def group_by_octave(a):
    """
    group array by octave
    in[note][time] == velocity
    out[octave][note][time] == velocity
    """
    chopped = a[21:105]
    return chopped.reshape(SONG_SHAPE)

def ungroup_by_octave(a):
    a = a.reshape((84, TOTAL_OUTPUT_TIME_LENGTH))
    a = np.concatenate((np.zeros((21, TOTAL_OUTPUT_TIME_LENGTH), dtype=np.float32),
                        a,
                        np.zeros((23, TOTAL_OUTPUT_TIME_LENGTH), dtype=np.float32)))
    return a


def res_to_midi(a):
    a = ungroup_by_octave(a)
    piano = ChannelArray(2, name='piano', program=4)
    piano.array = a
    track = numpy_to_midi_track(piano, 16)
    midi = mido.MidiFile(ticks_per_beat=16)

    midi.tracks.append(track)
    midi.save('generated.mid')


def generated_to_mid(filename):
    a = np.load(filename)
    assert a.shape == SONG_SHAPE
    res_to_midi(a)


def array_from_file(filename):
    """
    translates midi to array, returning None if error
    """
    print(filename)
    try:
        m = mido.MidiFile(filename)
        channels = numpy_from_midi(m)
        channel = map_to_one_channel(channels)
        res = group_by_octave(channel.array)
        res = res.tobytes()
    except (ValueError, IndexError,
            mido.midifiles.meta.KeySignatureError,
            EOFError, IOError,
            KeyError,):
        return None
    else:
        return res


def write_dataset(filename, data_list):
    ds = tf.data.Dataset.from_tensor_slices(data_list)
    recordwriter = tf.data.experimental.TFRecordWriter(filename)
    recordwriter.write(ds)


def parse_midis_and_save_batches(filenames_it):
    p = Pool(5)
    batch = []
    batch_size = 1024
    batch_num = 0
    for a in p.imap_unordered(array_from_file, filenames_it, 8):
        if a is None:
            continue
        batch.append(a)
        if len(batch) >= batch_size:
            print('-' * 20, 'save')
            write_dataset('datarecords/part%d.tfrecords' % batch_num, batch)
            batch_num += 1
            batch = []
    write_dataset('datarecords/part%d.tfrecords' % batch_num, batch)


def get_midi_names(dirname):
    res = (list(Path(dirname).glob('**/*.mid')) +
           list(Path(dirname).glob('**/*.MID')))
    random.shuffle(res)
    return res


def numpy_from_bytes(a):
    a = a.numpy()
    a = np.frombuffer(a, dtype=np.float32)
    return a.reshape(SONG_SHAPE)


def numpy_from_bytes_fn(a):
    return tf.py_function(numpy_from_bytes, inp=[a], Tout=tf.float32)


def get_dataset(files):
    dataset = (tf.data.TFRecordDataset(files)
               .map(numpy_from_bytes_fn, num_parallel_calls=AUTOTUNE)
               .cache('dataset_cache')
               .batch(64)
               .prefetch(4))
    return dataset


if __name__ == '__main__':
    # files = get_midi_names('/home/yoavm448/Downloads/torrents/videogame')
    # parse_midis_and_save_batches(files)
    files = list(map(str, Path('datarecords').glob('*.tfrecords')))
    print(len(files))
    if len(files) == 0:
        midi_files = get_midi_names('/home/yoavm448/Downloads/torrents/videogame')
        parse_midis_and_save_batches(midi_files)
    files = list(map(str, Path('datarecords').glob('*.tfrecords')))
    print('aoeu')
    ds = get_dataset(files)
    print(ds)
    for i in range(4):
        start = time.time()
        for x in ds:
            pass
        stop = time.time()
        print('time for epoch %d:, %3.2f'%(i+1, stop-start))

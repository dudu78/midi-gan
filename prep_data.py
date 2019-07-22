#!/usr/bin/env python

import mido
from midi_to_numpy import *
import numpy as np
import tensorflow as tf
from itertools import chain
from pathlib import Path

SONG_SHAPE = (12, 7, TOTAL_OUTPUT_TIME_LENGTH, 1)
dirname = 'videogame'
AUTOTUNE = tf.data.experimental.AUTOTUNE


def group_by_octave(a):
    """
    group array by octave
    in[note][time] == velocity
    out[octave][note][time] == velocity
    """
    chopped = a[21:105]
    return chopped.reshape(SONG_SHAPE)


def array_from_file(filename):
    """
    translates midi to array, returning None if error
    """
    filename = filename.numpy()
    # for x in filename:
        # print(x)
    try:
        m = mido.MidiFile(filename)
        channels = numpy_from_midi(m)
        channel = map_to_one_channel(channels)
        res = group_by_octave(channel.array)
    except:
        return np.full(SONG_SHAPE, -1, dtype=np.float32)
    else:
        return res


def array_from_file_fn(filename):
    return tf.py_function(array_from_file, inp=[filename], Tout=tf.float32)


def get_midi_names(dirname=dirname):  # unused
    return map(str, chain(Path(dirname).glob('**/*.mid'),
                          Path(dirname).glob('**/*.MID')))


def filter_inapropriate(a):
    if a.numpy().flatten()[0] == -1:
        return False
    return True


def filter_inapropriate_fn(a):
    return tf.py_function(filter_inapropriate, inp=[a], Tout=tf.bool)


def get_dataset():
    ds = tf.data.Dataset.list_files(dirname + '/**/*.mid')\
                        .shuffle(30000)\
                        .map(array_from_file_fn, num_parallel_calls=AUTOTUNE)\
                        .filter(filter_inapropriate_fn)\
                        .batch(64)\
                        .prefetch(1)\
                        #.cache(filename='dataset_cache')
    return ds


if __name__ == '__main__':
    ds = get_dataset()
    print('++ in for')
    for x in ds:
        print('+ for')
        print(x.shape)
        break

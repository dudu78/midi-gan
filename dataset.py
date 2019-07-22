#!/usr/bin/env python

import numpy as np
from pathlib import Path
from midi_to_numpy import *
import tensorflow as tf
import time


def song_gen(datapart_dir='dataparts'):
    for filename in Path(datapart_dir).glob('*.npy'):
        a = np.load(filename)
        assert a.shape[1:] == (7, 12, TOTAL_OUTPUT_TIME_LENGTH)
        for song in a:
            yield np.expand_dims(a, -1)


def get_dataset():
    return tf.data.Dataset\
                  .from_generator(song_gen, tf.float32)\
                  .batch(256)\
                  .cache('dataset_cache')


if __name__ == '__main__':
    print(tf.__version__)
    dataset = get_dataset()
    for batch in dataset:
        print(batch.shape)
        time.sleep(1)

#!/usr/bin/env python3.6
#-*-coding=utf-8-*-


import string
import numpy as np
import keras

from keras.datasets import imdb
#from keras.layers import preprocessing
from keras.preprocessing.sequence import pad_sequences

max_features = 10000
maxlen = 20

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train1 = pad_sequences(x_train, maxlen = maxlen)
x_test1 = pad_sequences(x_test, maxlen = maxlen)
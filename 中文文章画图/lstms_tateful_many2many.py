#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 13:51:57 2019
@author: jack
"""

# Stateful LSTM to learn one-char to one-char mapping
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
from keras.layers import TimeDistributed

# fix random seed for reproducibility
numpy.random.seed(7)
# define the raw dataset
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# create mapping of characters to integers (0-25) and the reverse
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
# prepare the dataset of input to output pairs encoded as integers
seq_length = 3
dataX = []
dataY = []
for i in range(0, len(alphabet) - 2*seq_length, 1):
	seq_in = alphabet[i:i + seq_length]
	seq_out = alphabet[i + seq_length:i + seq_length*2]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append([char_to_int[char] for char in seq_out])
	print(seq_in, '->', seq_out)
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (len(dataX), seq_length, 1))
# normalize
X = X / float(len(alphabet))
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# create and fit the model
batch_size = 1
model = Sequential()
model.add(LSTM(50, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True,return_sequences=True))
model.add(TimeDistributed(Dense(y.shape[2], activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
for i in range(30):
	model.fit(X, y, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
	model.reset_states()
    

res = model.predict(X,batch_size=batch_size,verbose=0)

"""
X.shape
Out[2]: (20, 3, 1)

y.shape
Out[3]: (20, 3, 25)

res.shape
Out[4]: (20, 3, 25)

"""


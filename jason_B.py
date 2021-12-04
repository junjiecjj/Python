from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LSTM
# prepare sequence
length = 100
time_step = 5
batch_size1 = 4
batch_size2 = 3
sample_num = int(length/time_step)
seq = array([i/float(length) for i in range(length)])
X = seq.reshape(sample_num, time_step, 1)
y = seq.reshape(sample_num, time_step, 1)
# define LSTM configuration
n_neurons = 10
n_batch = 1
n_epoch = 100
# create LSTM
model = Sequential()
model.add(LSTM(n_neurons, input_shape=(time_step, 1), return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
# train LSTM
model.fit(X, y, epochs=n_epoch, batch_size=batch_size1, verbose=2)
# evaluate
result = model.predict(X, batch_size=batch_size2, verbose=2)

"""
How to Use the TimeDistributed Layer for Long Short-Term Memory Networks in Python
中的最后一个案例

LSTM many2many,  stateful = false

"""

from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LSTM
import numpy as np
import matplotlib.pyplot as plt
# prepare sequence
length = 20*5*3
time_step = 5
batch_size1 = 4
batch_size2 = 3
sample_num = int(length/(time_step*3))
seq = array([i/float(length) for i in range(length)])
X = seq.reshape(sample_num, time_step, 3)  #(20,5,3)
train_x,train_y = X[:10], X[:10,:,-1].reshape(10,time_step,1)
test_x,test_y = X[3:], X[3:,:,-1].reshape(-1,time_step,1)
#test_x = test_x.reshape(17*5, 3)
#test_y = test_y.reshape(17*5, 1)

#train_x.shape=(10, 5, 3), train_y.shape=(10, 5, 1).
#test_x.shape=(17, 5, 3), test_y.shape=(17, 5, 1).
print("train_x.shape=%s, train_y.shape=%s."%(str(train_x.shape),str(train_y.shape)))
print("test_x.shape=%s, test_y.shape=%s."%(str(test_x.shape),str(test_y.shape)))
y = seq.reshape(sample_num, time_step, 3)
# define LSTM configuration
n_neurons = 10
n_epoch = 100
# create LSTM
model = Sequential()
model.add(LSTM(n_neurons, input_shape=(time_step, 3), return_sequences=True))
model.add(TimeDistributed(Dense(train_y.shape[2])))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
# train LSTM
his=model.fit(train_x, train_y, epochs=n_epoch, validation_data=(test_x,test_y),batch_size=batch_size1, verbose=2)
print(his.history.keys())
plt.plot(np.arange(n_epoch),his.history['loss'],'r')
plt.plot(np.arange(n_epoch),his.history['val_loss'],'b')
# evaluate
result = model.predict(test_x, batch_size=batch_size2, verbose=2)
print("result.shape=%s"%str(result.shape))
#result.shape = (17, 5, 1)

pred_y = result.reshape(-1,1)
real_y = test_y.reshape(-1,1)
fig=plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(pred_y.shape[0]).reshape(-1,1),pred_y,'r',label='pred')
ax.plot(np.arange(pred_y.shape[0]).reshape(-1,1),real_y,'b',label = 'real')
plt.legend(loc='best')
plt.show()


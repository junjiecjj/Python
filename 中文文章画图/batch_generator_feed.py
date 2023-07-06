#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 12:46:06 2019

@author: jack

这是查看某一炮的pcrl01,lmtipref,dfsdev,aminor的文件
"""
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import LSTM


#a = np.arange(12,dtype=np.int32).reshape(6,2)
#a = np.hstack([a,np.arange(6,dtype=np.int32).reshape(-1,1)])

#np.savetxt('./tttt.txt',a)

A = np.arange(10*3*2).reshape(10,3,2)

b  =np.array([1,2,3,4,5,6,7,8,9,10])

count =1



def generate_arrays_from_file(path,batch_size):
    global count
    n = 0
    while 1:
        print("count:"+str(count),'n='+str(n))
        datas = np.loadtxt(path,delimiter=' ',dtype="int")
        x = A[n*batch_size:(n+1)*batch_size]
        y = b[n*batch_size:(n+1)*batch_size]
        #x = datas[:,:2]
        #y = datas[:,2:]
        print("count = %d, x.shape=%s,y.shape=%s."%(count,x.shape,y.shape))
        #print('x = \n',x,'\n','y = \n',y)

        count = count+1
        yield (x,y)
        n = n+1
        if n == 6//3:
            n = 0
        #count = count+1
    return



x_valid = np.array([[[1,2],[2,3],[3,4]],[[ 6,  7],[ 8,  9],[10, 11]]])
y_valid = np.array([0,1])


model = Sequential()
model.add(LSTM(units=100,input_shape=(3,2)))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='mse',
              optimizer='sgd',
              metrics=['mae'])
model.fit_generator(generate_arrays_from_file("./tttt.txt",5),\
                    steps_per_epoch=2,epochs=1,\
                    max_queue_size=1,\
                    use_multiprocessing = True,\
                    validation_data=(x_valid, y_valid),workers=1)




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation


#https://www.jianshu.com/p/5d6d5aac4dbd

def load_data(file_name, sequence_length=10, split=0.8):
    df = pd.read_csv(file_name, sep=',', usecols=[1])
    data_all = np.array(df).astype(float)
    scaler = MinMaxScaler()
    data_all = scaler.fit_transform(data_all)
    data = []
    for i in range(len(data_all) - sequence_length - 1):
        data.append(data_all[i: i + sequence_length + 1])
    reshaped_data = np.array(data).astype('float64')
    np.random.shuffle(reshaped_data)
    # 对x进行统一归一化，而y则不归一化
    x = reshaped_data[:, :-1]
    y = reshaped_data[:, -1]
    split_boundary = int(reshaped_data.shape[0] * split)
    train_x = x[: split_boundary]
    test_x = x[split_boundary:]

    train_y = y[: split_boundary]
    test_y = y[split_boundary:]

    return train_x, train_y, test_x, test_y, scaler


def build_model():
    # input_dim是输入的train_x的最后一个维度，train_x的维度为(n_samples, time_steps, input_dim)
    model = Sequential()
    model.add(LSTM(units = 50, input_shape=(10,1),  return_sequences=True))
    print(model.layers)
    model.add(LSTM(units =100, return_sequences=False))
    model.add(Dense(units =1))
    model.add(Activation('linear'))
    #因为最后一层只有一个神经元，所以为many-to-one的模型
    model.compile(loss='mse', optimizer='rmsprop')
    return model


def train_model(train_x, train_y, test_x, test_y):
    model = build_model()
    
    try:
        model.fit(train_x, train_y, batch_size=512, epochs=30, validation_split=0.1)
        predict = model.predict(test_x)  # predict_y.shape = (27,1)
        predict = np.reshape(predict, (predict.size, ))
    except KeyboardInterrupt:
        print(predict)
        print(test_y)
    print(predict)
    print(test_y)
    try:
        fig = plt.figure(1)
        plt.plot(predict, 'r:')
        plt.plot(test_y, 'g-')
        plt.legend(['predict', 'true'])
    except Exception as e:
        print(e)
    print(model.summary())
    return predict, test_y


#if __name__ == '__main__':
train_x, train_y, test_x, test_y, scaler = load_data('international-airline-passengers.csv')
"""
train_x.shape, train_y.shape, test_x.shape, test_y.shape
Out[12]: ((106, 10, 1), (106, 1), (27, 10, 1), (27, 1))
"""

train_x1 = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))  #(106, 10, 1)
test_x1 = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))  #(27, 10, 1)

predict_y, test_y = train_model(train_x1, train_y, test_x1, test_y)
predict_y1 = scaler.inverse_transform([[i] for i in predict_y])
test_y1 = scaler.inverse_transform(test_y)
"""
predict_y.shape, test_y.shape, predict_y1.shape,test_y1.shape
Out[20]: ((27,), (27, 1), (27, 1), (27, 1))
"""

fig2 = plt.figure(2)
plt.plot(predict_y1, 'g:')
plt.plot(test_y1, 'r-')
plt.show()


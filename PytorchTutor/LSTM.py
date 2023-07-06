#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 19:17:43 2022

@author: jack

https://mp.weixin.qq.com/s?__biz=MzA5MzUxMzg5NA==&mid=2453846790&idx=1&sn=6b4ab17603e8fd7d13ca83f0b4b01d36&chksm=87eaaacfb09d23d9fcddbb7def13c64fe3a5244c114f076a92dd0c8f499ada8bbdf152d29fdf&mpshare=1&scene=24&srcid=0329iDFoIozV3EbXvpQQlYci&sharer_sharetime=1648539034087&sharer_shareid=8d8081f5c3018ad4fbee5e86ad64ec5c&exportkey=AbWZVNTz0io%2Bhi0%2Bj8a%2Bzdk%3D&acctmode=0&pass_ticket=146ss%2BlmDqrmXaN16A3PP0abDTvpLOnTw2EFkc6v8OK8141SELQZOf%2BbYOJluu8o&wx_header=0#rd

保姆级教程，用 PyTorch 构建第一个神经网络
Python开发精选 2022-03-29 12:05

"""






# 导入相关模块
import torch

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
from torch import nn, optim

 


sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 14, 6
register_matplotlib_converters()

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

#探索性数据分析
df = pd.read_csv('/home/jack/公共的/MLData/time_series_covid19_confirmed_global.csv')
print("df.head() = {}\n".format(df.head()))

df = df.iloc[:, 4:]
print("df.head() = {}\n".format(df.head()))


print("df.isnull().sum().sum() = {}\n".format(df.isnull().sum().sum()))

daily_cases = df.sum(axis=0)
daily_cases.index = pd.to_datetime(daily_cases.index)
daily_cases.head()
plt.plot(daily_cases)
plt.title("Cumulative daily cases");

daily_cases = daily_cases.diff().fillna(daily_cases[0]).astype(np.int64)
daily_cases.head()
plt.plot(daily_cases)
plt.title("Daily cases");

print("daily_cases.shape = {}\n".format(daily_cases.shape))

#数据预处理
test_data_size = 155

train_data = daily_cases[:-test_data_size]
test_data = daily_cases[-test_data_size:]

print("train_data.shape  = {}\n".format(train_data.shape)) # train_data.shape  = (644,)
print("test_data.shape  = {}\n".format(test_data.shape))   # test_data.shape  = (155,)


scaler = MinMaxScaler()
scaler = scaler.fit(np.expand_dims(train_data, axis=1))

train_data = scaler.transform(np.expand_dims(train_data, axis=1))
test_data = scaler.transform(np.expand_dims(test_data, axis=1))

print("  train_data.shape  = {}".format(train_data.shape))  # train_data.shape  = (644, 1)
print("  test_data.shape  = {}".format(test_data.shape))    # test_data.shape  = (155, 1)


def create_sequences(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

seq_length = 7
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

print("1.  X_train.shape  = {}, y_train.shape = {} \n".format(X_train.shape,y_train.shape)) # X_train.shape  = (636, 7, 1), y_train.shape = (636, 1) 
print("1.  X_test.shape  = {}, y_test.shape = {} \n".format(X_test.shape,y_test.shape))     #  X_test.shape  = (147, 7, 1), y_test.shape = (147, 1) 


X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()
print("2.  X_train.shape  = {}, y_train.shape = {} \n".format(X_train.shape,y_train.shape))
# X_train.shape  = torch.Size([636, 7, 1]), y_train.shape = torch.Size([636, 1]) 
print("2.  X_test.shape  = {}, y_test.shape = {} \n".format(X_test.shape,y_test.shape))
# X_test.shape  = torch.Size([147, 7, 1]), y_test.shape = torch.Size([147, 1]) 

#建立模型

class CoronaVirusPredictor(nn.Module):

    def __init__(self, n_features, n_hidden, seq_len, n_layers=2):
        super(CoronaVirusPredictor, self).__init__()

        self.n_hidden = n_hidden   #  n_hidden = 512
        self.seq_len = seq_len     # seq_len = 7
        self.n_layers = n_layers  # n_layers = 2

        self.lstm = nn.LSTM(
          input_size=n_features,  # 1
          hidden_size=n_hidden,  # 512
          num_layers=n_layers,  # 2
          dropout=0.5,
          #batch_first= True
        )

        self.linear = nn.Linear(in_features=n_hidden, out_features=1)

    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden)
        )

    def forward(self, sequences):
        # print("sequences.shape = {}\n".format(sequences.shape)) #sequences.shape = torch.Size([636, 7, 1])
        
        lstm_out, self.hidden = self.lstm( sequences.view(len(sequences), self.seq_len, -1), self.hidden )
        #print("sequences.view(len(sequences), self.seq_len, -1).shape = {}\n".format(sequences.view(len(sequences), self.seq_len, -1).shape))
        # sequences.view(len(sequences), self.seq_len, -1).shape = torch.Size([636, 7, 1])
        
        #print("len(sequences) = {}\n".format(len(sequences)))  # len(sequences) = 636
        #print("lstm_out.shape = {}\n".format(lstm_out.shape))  #lstm_out.shape = torch.Size([636, 7, 512])
        
        last_time_step = lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
        print("lstm_out.view(self.seq_len, len(sequences), self.n_hidden).shape = {}\n".format(lstm_out.view(self.seq_len, len(sequences), self.n_hidden).shape))
        # lstm_out.view(self.seq_len, len(sequences), self.n_hidden).shape = torch.Size([7, 636, 512])
        
        
        print("last_time_step.shape = {}\n".format(last_time_step.shape))
        #  last_time_step.shape = torch.Size([636, 512])
        
        y_pred = self.linear(last_time_step)
        print("y_pred.shape = {}\n".format(y_pred.shape))  # y_pred.shape = torch.Size([636, 1])
        return y_pred



#训练模型
def train_model( model,  train_data,  train_labels,  test_data=None,  test_labels=None):
    loss_fn = torch.nn.MSELoss(reduction='sum')

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 3

    train_hist = np.zeros(num_epochs)
    test_hist = np.zeros(num_epochs)

    for t in range(num_epochs):
        model.reset_hidden_state()
        y_pred = model(X_train)  #y_pred.shape = torch.Size([636, 1])
        print("2.............y_pred.shape = {}\n".format(y_pred.shape))
        loss = loss_fn(y_pred.float(), y_train)
        if test_data is not None:
            with torch.no_grad():
                y_test_pred = model(X_test)
                test_loss = loss_fn(y_test_pred.float(), y_test)
            test_hist[t] = test_loss.item()

            if t % 10 == 0:  
                print(f'Epoch {t} train loss: {loss.item()} test loss: {test_loss.item()}')
        elif t % 10 == 0:
            print(f'Epoch {t} train loss: {loss.item()}')

        train_hist[t] = loss.item()
        optimiser.zero_grad()  # 清零梯度缓存
        loss.backward()        # 反向传播误差
        optimiser.step()       # 更新参数

    return model.eval(), train_hist, test_hist


model = CoronaVirusPredictor(
  n_features=1, 
  n_hidden=512, 
  seq_len=seq_length,   # 7
  n_layers=1
)
model, train_hist, test_hist = train_model(
  model, 
  X_train,  # torch.Size([636, 7, 1])
  y_train,  # torch.Size([636, 1])
  X_test,   # torch.Size([147, 7, 1])
  y_test    # torch.Size([147, 1])
)
"""
model
Out[35]: 
CoronaVirusPredictor(
  (lstm): LSTM(1, 512, num_layers=2, dropout=0.5)
  (linear): Linear(in_features=512, out_features=1, bias=True)
)

train_hist.shape
Out[36]: (3,)

test_hist.shape
Out[37]: (3,)
"""



plt.plot(train_hist, label="Training loss")
plt.plot(test_hist, label="Test loss")
plt.ylim((0, 100))
plt.legend();





#预测未来几天的病例

with torch.no_grad():
    test_seq = X_test[:1]
    preds = []
    for _ in range(len(X_test)):
        y_test_pred = model(test_seq)
        pred = torch.flatten(y_test_pred).item()
        preds.append(pred)
        new_seq = test_seq.numpy().flatten()
        new_seq = np.append(new_seq, [pred])
        new_seq = new_seq[1:]
        test_seq = torch.as_tensor(new_seq
                   ).view(1, seq_length, 1).float()



true_cases = scaler.inverse_transform(
    np.expand_dims(y_test.flatten().numpy(), axis=0)
).flatten()

predicted_cases = scaler.inverse_transform(
  np.expand_dims(preds, axis=0)
).flatten()



plt.plot(
  daily_cases.index[:len(train_data)], 
  scaler.inverse_transform(train_data).flatten(),
  label='Historical Daily Cases')

plt.plot(
  daily_cases.index[len(train_data):len(train_data) + len(true_cases)], 
  true_cases,
  label='Real Daily Cases')

plt.plot(
  daily_cases.index[len(train_data):len(train_data) + len(true_cases)], 
  predicted_cases, 
  label='Predicted Daily Cases')

plt.legend();




#使用所有数据来训练
scaler = MinMaxScaler()

scaler = scaler.fit(np.expand_dims(daily_cases, axis=1))
all_data = scaler.transform(np.expand_dims(daily_cases, axis=1))
all_data.shape


#预处理和训练步骤相同。
X_all, y_all = create_sequences(all_data, seq_length)

X_all = torch.from_numpy(X_all).float()
y_all = torch.from_numpy(y_all).float()

model = CoronaVirusPredictor(
  n_features=1, 
  n_hidden=512, 
  seq_len=seq_length, 
  n_layers=2
)
model, train_hist, _ = train_model(model, X_all, y_all)



#预测未来病例
DAYS_TO_PREDICT = 12

with torch.no_grad():
    test_seq = X_all[:1]
    preds = []
    for _ in range(DAYS_TO_PREDICT):
        y_test_pred = model(test_seq)
        pred = torch.flatten(y_test_pred).item()
        preds.append(pred)
        new_seq = test_seq.numpy().flatten()
        new_seq = np.append(new_seq, [pred])
        new_seq = new_seq[1:]
        test_seq = torch.as_tensor(new_seq).view(1, seq_length, 1).float()



#和以前一样，我们将逆缩放器变换。
predicted_cases = scaler.inverse_transform(
  np.expand_dims(preds, axis=0)
).flatten()





#要使用历史和预测案例创建一个很酷的图表，我们需要扩展数据框的日期索引。
daily_cases.index[-1]


predicted_index = pd.date_range(
  start=daily_cases.index[-1],
  periods=DAYS_TO_PREDICT + 1,
  closed='right'
)

predicted_cases = pd.Series(
  data=predicted_cases,
  index=predicted_index
)

plt.plot(predicted_cases, label='Predicted Daily Cases')
plt.legend();

plt.plot(daily_cases, label='Historical Daily Cases')
plt.plot(predicted_cases, label='Predicted Daily Cases')
plt.legend();



"""
print("2.  X_train.shape  = {}, y_train.shape = {} \n".format(X_train.shape,y_train.shape))
# X_train.shape  = torch.Size([636, 7, 1]), y_train.shape = torch.Size([636, 1]) 
print("2.  X_test.shape  = {}, y_test.shape = {} \n".format(X_test.shape,y_test.shape))
# X_test.shape  = torch.Size([147, 7, 1]), y_test.shape = torch.Size([147, 1]) 

def forward(self, sequences):
    # print("sequences.shape = {}\n".format(sequences.shape)) #sequences.shape = torch.Size([636, 7, 1])
        lstm_out, self.hidden = self.lstm(
          sequences.view(len(sequences), self.seq_len, -1),
          self.hidden
        )
        last_time_step = \
          lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
        y_pred = self.linear(last_time_step)
        return y_pred
    
    
lstm_out, self.hidden = self.lstm( sequences.view(len(sequences), self.seq_len, -1), self.hidden )
#print("sequences.view(len(sequences), self.seq_len, -1).shape = {}\n".format(sequences.view(len(sequences), self.seq_len, -1).shape))
# sequences.view(len(sequences), self.seq_len, -1).shape = torch.Size([636, 7, 1])

下面介绍一下输入数据的维度要求(batch_first=False)。输入数据需要按如下形式传入input, (h_0,c_0)
input: 输入数据，即上面例子中的一个句子（或者一个batch的句子），其维度形状为 (seq_len, batch, input_size)
seq_len: 句子长度，即单词数量，这个是需要固定的。当然假如你的一个句子中只有2个单词，但是要求输入10个单词，这个时候可以用torch.nn.utils.rnn.pack_padded_sequence()，或者torch.nn.utils.rnn.pack_sequence()来对句子进行填充或者截断。
batch：就是你一次传入的句子的数量
input_size: 每个单词向量的长度，这个必须和你前面定义的网络结构保持一致
h_0：维度形状为 (num_layers * num_directions, batch, hidden_size)。
结合下图应该比较好理解第一个参数的含义num_layers * num_directions，即LSTM的层数乘以方向数量。这个方向数量是由前面介绍的bidirectional决定，如果为False,则等于1；反之等于2。
batch：同上
hidden_size: 隐藏层节点数
c_0：维度形状为 (num_layers * num_directions, batch, hidden_size)，各参数含义和h_0类似。

"""




































































































































































































































































































































































































































































































































































































































































































































































































#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:28:33 2022

@author: jack

https://mp.weixin.qq.com/s?__biz=MzA5MzUxMzg5NA==&mid=2453846790&idx=1&sn=6b4ab17603e8fd7d13ca83f0b4b01d36&chksm=87eaaacfb09d23d9fcddbb7def13c64fe3a5244c114f076a92dd0c8f499ada8bbdf152d29fdf&mpshare=1&scene=24&srcid=0329iDFoIozV3EbXvpQQlYci&sharer_sharetime=1648539034087&sharer_shareid=8d8081f5c3018ad4fbee5e86ad64ec5c&exportkey=AbWZVNTz0io%2Bhi0%2Bj8a%2Bzdk%3D&acctmode=0&pass_ticket=146ss%2BlmDqrmXaN16A3PP0abDTvpLOnTw2EFkc6v8OK8141SELQZOf%2BbYOJluu8o&wx_header=0#rd

保姆级教程，用 PyTorch 构建第一个神经网络
Python开发精选 2022-03-29 12:05

"""

#导入相关模块
import torch

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch import nn, optim
import torch.nn.functional as F
#from ann_visualizer.visualize import ann_viz
import pretty_errors

# 【重点】进行配置
pretty_errors.configure(
    separator_character = '*',
    filename_display    = pretty_errors.FILENAME_EXTENDED,
    filename_color = pretty_errors.BRIGHT_YELLOW,
    line_number_first   = True,
    display_link        = True,
    lines_before        = 5,
    lines_after         = 2,
    line_color          = pretty_errors.RED + '> ' + pretty_errors.default_config.line_color,
    code_color          = '  ' + pretty_errors.default_config.line_color,
    header_color        = 'blue'
)


#构建神经网络
class Net(nn.Module):

    def __init__(self, n_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 5)
        self.fc2 = nn.Linear(5, 3)
        self.fc3 = nn.Linear(3, 1)

    def forward(self, x):
        #print("x.shape = {}\n".format(x.shape)) #x.shape = torch.Size([99751, 4])
        
        x = F.relu(self.fc1(x))
        #print("x111.shape = {}\n".format(x.shape)) #x111.shape = torch.Size([99751, 5])
        
        x = F.relu(self.fc2(x))
        
        #print("x222.shape = {}\n".format(x.shape)) #x222.shape = torch.Size([99751, 3])
        return torch.sigmoid(self.fc3(x))

#寻找最优参数
def calculate_accuracy(y_true, y_pred):
    predicted = y_pred.ge(.5).view(-1)
    return (y_true == predicted).sum().float() / len(y_true)



def will_it_rain(rainfall, humidity, rain_today, pressure):
    t = torch.as_tensor([rainfall, humidity, rain_today, pressure]).float().to(device)
    output = net(t)
    return output.ge(0.5).item()


def round_tensor(t, decimal_places=3):
    return round(t.item(), decimal_places)





#模型预测
def will_it_rain(rainfall, humidity, rain_today, pressure):
     """
     rainfall=10
     humidity=10
     rain_today=True
     pressure=2
     
     torch.as_tensor([rainfall, humidity, rain_today, pressure]).float()
     Out[257]: tensor([10., 10.,  1.,  2.])
     """
     t = torch.as_tensor([rainfall, humidity, rain_today, pressure]).float().to(device)
     output = net(t)
     return output.ge(0.5).item()



sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 6
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)








#数据集
df = pd.read_csv('/home/jack/公共的/MLData/weatherAUS.csv')
print("df.head() = {}\n".format(df.head()))
print("df.shape = {}\n".format(df.shape))#df.shape = (145460, 23)

#数据预处理
cols = ['Rainfall', 'Humidity3pm', 'Pressure9am', 'RainToday', 'RainTomorrow']
df = df[cols]

#特征转换
df['RainToday'].replace({'No': 0, 'Yes': 1}, inplace = True)
df['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace = True)

#缺失值处理
df = df.dropna(how='any')
print("df.head() = {}\n".format(df.head()))

#样本不平衡处理
sns.countplot(df.RainTomorrow);


#从结果看，下雨次数明显比不下雨次数要少很多。再通过具体定量计算正负样本数。
print("df.RainTomorrow.value_counts() / df.shape[0] = \n{}\n".format(df.RainTomorrow.value_counts() / df.shape[0]))

#样划分训练集和测试集

X = df[['Rainfall', 'Humidity3pm', 'RainToday', 'Pressure9am']]
y = df[['RainTomorrow']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)



#数据类型转换
X_train = torch.from_numpy(X_train.to_numpy()).float()
y_train = torch.squeeze(torch.from_numpy(y_train.to_numpy()).float())

X_test = torch.from_numpy(X_test.to_numpy()).float()
y_test = torch.squeeze(torch.from_numpy(y_test.to_numpy()).float())

print("X_train.shape = {}, y_train.shape = {}\n".format(X_train.shape, y_train.shape))
print("X_test.shape = {}, y_test.shape = {}\n".format(X_test.shape, y_test.shape))
# X_train.shape = torch.Size([99751, 4]), y_train.shape = torch.Size([99751])
# X_test.shape = torch.Size([24938, 4]), y_test.shape = torch.Size([24938])



#可视化神经元
net = Net(X_train.shape[1])
#Build your model here
#ann_viz(net, view=True)


#损失函数
criterion = nn.BCELoss()

#优化器
optimizer = optim.Adam(net.parameters(), lr=0.001)

#在 GPU 上计算
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

X_train = X_train.to(device)
y_train = y_train.to(device)

X_test = X_test.to(device)
y_test = y_test.to(device)

net = net.to(device)
criterion = criterion.to(device)


#所有的模块都准备好了，我们可以开始训练我们的模型了。
for epoch in range(1000):    
    y_pred = net(X_train)
    print(f"1  y_pred.shape = {y_pred.shape}") # y_pred.shape = torch.Size([99751, 1])
    y_pred = torch.squeeze(y_pred)
    print(f"2  y_pred.shape = {y_pred.shape}") # y_pred.shape = torch.Size([99751])
    train_loss = criterion(y_pred, y_train)
    
    if epoch % 100 == 0:
        train_acc = calculate_accuracy(y_train, y_pred)

        y_test_pred = net(X_test)
        y_test_pred1 = torch.squeeze(y_test_pred)

        test_loss = criterion(y_test_pred1, y_test)
        test_acc = calculate_accuracy(y_test, y_test_pred1)
        print(f'''epoch {epoch}
              Train set - loss: {round_tensor(train_loss)}, accuracy: {round_tensor(train_acc)}
              Test  set - loss: {round_tensor(test_loss)}, accuracy: {round_tensor(test_acc)}
              ''')
    
    optimizer.zero_grad()  # 清零梯度缓存
    train_loss.backward() # 反向传播误差
    optimizer.step()  # 更新参数




MODEL_PATH = 'model.pth'  # 后缀名为 .pth
torch.save(net, MODEL_PATH) # 直接使用torch.save()函数即可




net = torch.load(MODEL_PATH)

classes = ['No rain', 'Raining']

y_pred = net(X_test)
y_pred1 = y_pred.ge(.5).view(-1).cpu()  #
y_test = y_test.cpu()

print(classification_report(y_test, y_pred1, 
                            target_names=classes))


cm = confusion_matrix(y_test, y_pred1)
df_cm = pd.DataFrame(cm, index=classes, columns=classes)

hmap = sns.heatmap(df_cm, annot=True, fmt="d")
hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
plt.ylabel('True label')
plt.xlabel('Predicted label');


#模型预测
will_it_rain(rainfall=10, humidity=10, 
             rain_today=True, pressure=2)




will_it_rain(rainfall=0, humidity=1, 
             rain_today=False, pressure=100)


































































































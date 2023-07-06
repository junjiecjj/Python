#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 18:41:02 2022

@author: jack

https://lulaoshi.info/machine-learning/neural-network/pytorch-kaggle-house-prices

"""
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F


# 选出非空列
def get_cols_with_no_nans(df,col_type):
    '''
    Arguments :
    df : The dataframe to process
    col_type :
          num : to only get numerical columns with no nans
          no_num : to only get nun-numerical columns with no nans
          all : to get any columns with no nans
    '''
    if (col_type == 'num'):
        predictors = df.select_dtypes(exclude=['object'])
    elif (col_type == 'no_num'):
        predictors = df.select_dtypes(include=['object'])
    elif (col_type == 'all'):
        predictors = df
    else :
        print('Error : choose a type (num, no_num, all)')
        return 0
    cols_with_no_nans = []
    for col in predictors.columns:
        if not df[col].isnull().any():
            cols_with_no_nans.append(col)
    return cols_with_no_nans

# 对分类特征进行One-Hot编码
def oneHotEncode(df,colNames):
    for col in colNames:
        if( df[col].dtype == np.dtype('object')):
            # pandas.get_dummies 可以对分类特征进行One-Hot编码
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df,dummies], axis=1)

            # drop the encoded column
            df.drop([col], axis = 1 , inplace=True)
    return df

#另一种构建神经网络的方式是继承nn.Module类，我们将子类起名为Net类。__init__()方法为Net类的构造函数，用来初始化神经网络各层的参数；forward()也是我们必须实现的方法，主要用来实现神经网络的前向传播过程。
class Net(nn.Module):

    def __init__(self, features):
        super(Net, self).__init__()

        self.linear_relu1 = nn.Linear(features, 128)
        self.linear_relu2 = nn.Linear(128, 256)
        self.linear_relu3 = nn.Linear(256, 256)
        self.linear_relu4 = nn.Linear(256, 256)
        self.linear5 = nn.Linear(256, 1)

    def forward(self, x):

        y_pred = self.linear_relu1(x)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear_relu2(y_pred)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear_relu3(y_pred)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear_relu4(y_pred)
        y_pred = nn.functional.relu(y_pred)

        y_pred = self.linear5(y_pred)
        return y_pred



train_data_path ='train.csv'
train = pd.read_csv(train_data_path)
print(f"train.shape = {train.shape}")#train.shape = (1460, 81)
print(f"train.describe() = \n{train.describe()}")
num_of_train_data = train.shape[0]


test_data_path ='test.csv'
test = pd.read_csv(test_data_path)
print(f"test.shape = {test.shape}") #test.shape = (1459, 80)


# 房价，要拟合的目标值
target = train.SalePrice  # (1460,)

# 输入特征，可以将SalePrice列扔掉
# inplace可选参数,如果手动设定为True（默认为False），那么原数组直接就被替换。也就是说，采用inplace=True之后，原数组名（如2和3情况所示）对应的内存值直接改变；
train.drop(['SalePrice'],axis = 1 , inplace = True)  # (1460, 80)

# 将train和test合并到一起，一块进行特征工程，方便预测test的房价
combined = train.append(test)
combined.reset_index(inplace=True)
combined.drop(['index', 'Id'], inplace=True, axis=1)





num_cols = get_cols_with_no_nans(combined, 'num')
cat_cols = get_cols_with_no_nans(combined, 'no_num')

# 过滤掉含有缺失值的特征
combined = combined[num_cols + cat_cols]

print(num_cols[:5])
print ('Number of numerical columns with no nan values: ',len(num_cols))
print(cat_cols[:5])
print ('Number of non-numerical columns with no nan values: ',len(cat_cols))

combined = oneHotEncode(combined,combined.columns)


# 训练数据集特征
train_features = torch.tensor(combined[:num_of_train_data].values, dtype=torch.float)
# 训练数据集目标
train_labels = torch.tensor(target.values, dtype=torch.float).view(-1, 1)
# 测试数据集特征
test_features = torch.tensor(combined[num_of_train_data:].values, dtype=torch.float)

print("train data size: ", train_features.shape)
print("label data size: ", train_labels.shape)
print("test data size: ", test_features.shape)
"""
train data size:  torch.Size([1460, 149])
label data size:  torch.Size([1460, 1])
test data size:  torch.Size([1459, 149])
"""



"""
我们已经定义好了一个神经网络的Net类，还要初始化一个Net类的对象实例model，
表示某个具体的模型。然后定义损失函数，这里使用MSELoss，MSELoss使用了均方误差（Mean Square Error）来衡量损失函数。
对于模型model的训练过程，这里使用Adam算法。Adam是优化算法中的一种，在很多场景中效率要优于SGD。
"""
model = Net(features=train_features.shape[1])

# 使用均方误差作为损失函数
criterion = nn.MSELoss(reduction='mean')

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


#接着，我们使用Adam算法进行多轮的迭代，更新模型model中的参数。这里对模型进行500轮的迭代。

losses = []

# 训练500轮
for t in range(500):
    y_pred = model(train_features)
    # print(f"y_pred.shape = {y_pred.shape}")  #y_pred.shape = torch.Size([1460, 1])
    loss = criterion(y_pred, train_labels)
    # print(t, loss.item())
    losses.append(loss.item())

    if torch.isnan(loss):
        break

    # 将模型中各参数的梯度清零。
    # PyTorch的backward()方法计算梯度会默认将本次计算的梯度与缓存中已有的梯度加和。
    # 必须在反向传播前先清零。
    optimizer.zero_grad()

    # 反向传播，计算各参数对于损失loss的梯度
    loss.backward()

    # 根据刚刚反向传播得到的梯度更新模型参数
    optimizer.step()


#我们可以使用模型对测试数据集进行预测，将得到的预测值保存成文件，提交到Kaggle上。
predictions = model(test_features).detach().numpy()
my_submission = pd.DataFrame({'Id':pd.read_csv('test.csv').Id,'SalePrice': predictions[:, 0]})
my_submission.to_csv('{}.csv'.format('./submission'), index=False)

























































































































































































































































































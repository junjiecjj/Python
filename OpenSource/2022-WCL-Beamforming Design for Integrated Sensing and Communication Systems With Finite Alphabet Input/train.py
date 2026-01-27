import numpy as np
import h5py
import torch
from model import *
import torch.utils.data  as Data
import matplotlib.pyplot as plt
import time
import scipy.io as scio
import os


BATCH_SIZE=500
LR = 0.0001

feature = h5py.File('train_data_tmp.mat')
F_data = feature['F_data_t'][:]
H_data = feature['H_data_t'][:]
F_data = np.transpose(F_data)
H_data = np.transpose(H_data)

F_data_tensor = torch.from_numpy(F_data)
F_data_tensor = F_data_tensor.to(torch.float32)
H_data_tensor = torch.from_numpy(H_data)
H_data_tensor = H_data_tensor.to(torch.float32)

F_data_tensor=(F_data_tensor+0.2500)*2
H_data_tensor=(H_data_tensor+3.5000)/7

train_data_x =H_data_tensor[:6000]
train_data_y =F_data_tensor[:6000]
validation_data_x =H_data_tensor[6000:6800]
validation_data_y =F_data_tensor[6000:6800]

torch_dataset = Data.TensorDataset(train_data_x,train_data_y)
loader = Data.DataLoader(dataset=torch_dataset,batch_size=BATCH_SIZE,shuffle=True)

net = Net(144,4*144,2*144,4*144,144)
print(net)
optimizer = torch.optim.Adam(net.parameters(),lr = LR)
loss_func = torch.nn.MSELoss()

loss_his=[]
acc_his=[]

start_time=time.time()
acc=1
num=0

for epoch in range(1000):
    for step, (batch_x, batch_y) in enumerate(loader):
        prediction = net(batch_x)
        loss = loss_func(prediction, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch%10 ==0:
        print('epoch: %d' % epoch)
        print('Loss = %.4f' % loss.data)
        print('time: %.4f' % (time.time()-start_time))
        loss_his.append(loss.data)

        y_out = net(validation_data_x)
        accuracy = loss_func(y_out, validation_data_y)
        acc_his.append(accuracy.data)

        if accuracy.data < acc:
            torch.save(net, 'mat.pkl')
            num = epoch
            acc = accuracy.data




plt.plot(loss_his, c='blue',  linestyle='-', label='Loss')
plt.plot(acc_his, c='red',  linestyle='-', label='Accuracy')
plt.legend(loc=1)
plt.show()

torch.save(net,'mat.pkl')







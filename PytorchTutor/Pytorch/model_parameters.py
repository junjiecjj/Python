#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 21:25:59 2022
@author: jack

验证模型保存时保存的.pt文件大小是否改变，以及随着训练的进行，模型的参数数值是否改变。结果表明：

1. 一旦模型确定，则模型的pt大小是确定的，而不管模型的参数怎么变。
2. 随着训练过程的持续，模型的参数一直在变。
3. 随着训练过程的推荐，冻结的那些层的参数不会改变。
"""

import sys,os
import torch
from torch.autograd import Variable


import torch.nn as nn
import imageio


import matplotlib
matplotlib.use('TkAgg')


import torch.optim as optim


#=============================================================================================================
#  AE based on cnn for MNIST
#=============================================================================================================
import copy
def Quantize(img, bits = 8):
    ###  1
    # Range = 2**bits
    # Img_max = torch.max(img)
    # img = img / Img_max
    # img = img * Range
    # img = img.round()
    # img = img / Range
    # img = img * Img_max

    ###  2
    img = img.detach()
    x_max = torch.max(img)
    x_tmp = copy.deepcopy(torch.div(img, x_max))

    # quantize
    x_tmp = copy.deepcopy(torch.mul(x_tmp, 256))
    x_tmp = copy.deepcopy(x_tmp.clone().type(torch.int))
    x_tmp = copy.deepcopy(x_tmp.clone().type(torch.float32))
    x_tmp = copy.deepcopy(torch.div(x_tmp, 256))

    img = copy.deepcopy(torch.mul(x_tmp, x_max))

    return img
# 以实际信号功率计算噪声功率，再将信号加上噪声。
def Awgn(x, snr = 3):
    if snr == None:
        return x
    SNR = 10.0**(snr/10.0)
    # signal_power = ((x**2)*1.0).mean()
    signal_power = (x*1.0).pow(2).mean()
    noise_power = signal_power/SNR
    noise_std = torch.sqrt(noise_power)
    #print(f"x.shape = {x.shape}, signal_power = {signal_power}, noise_power={noise_power}, noise_std={noise_std}")

    noise = torch.normal(mean = 0, std = float(noise_std), size = x.shape)
    return x + noise.to(x.device)


# https://blog.csdn.net/weixin_38739735/article/details/119013420
class Encoder_cnn_mnist(nn.Module):
    def __init__(self, encoded_space_dim):
        super(Encoder_cnn_mnist, self).__init__()
        ### Convolutional p
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear p
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )
    def forward(self, x):
        # print(f"1 x.shape = {x.shape}")
        # torch.Size([25, 1, 28, 28])
        x = self.encoder_cnn(x)
        # print(f"2 x.shape = {x.shape}")
        # torch.Size([25, 32, 3, 3])
        x = self.flatten(x)
        # print(f"3 x.shape = {x.shape}")
        # torch.Size([25, 288])
        x = self.encoder_lin(x)
        # print(f"4 x.shape = {x.shape}")
        # torch.Size([25, 4])
        return x


class Decoder_cnn_mnist(nn.Module):
    def __init__(self, encoded_space_dim):
        super(Decoder_cnn_mnist, self).__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2,  padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )
    def forward(self, x):
        # print(f"1 x.shape = {x.shape}")
        # 1 torch.Size([25, 4])
        x = self.decoder_lin(x)
        # print(f"2 x.shape = {x.shape}")
        # 2 x.shape = torch.Size([25, 288])
        x = self.unflatten(x)
        # print(f"3 x.shape = {x.shape}")
        # 3 x.shape = torch.Size([25, 32, 3, 3])
        x = self.decoder_conv(x)
        # print(f"4 x.shape = {x.shape}")
        # 4 x.shape = torch.Size([25, 1, 28, 28])
        # x = torch.sigmoid(x)
        x = torch.tanh(x)
        # print(f"5 x.shape = {x.shape}")
        # 5 x.shape = torch.Size([25, 1, 28, 28])
        return x

class AED_cnn_mnist(nn.Module):
    def __init__(self, encoded_space_dim = 100, snr  = 3, quantize = True):
        super(AED_cnn_mnist, self).__init__()
        self.snr = snr
        self.quantize = quantize
        self.encoder = Encoder_cnn_mnist(encoded_space_dim)
        self.decoder = Decoder_cnn_mnist(encoded_space_dim)

    def forward(self, img):
        # print(f"img.shape = {img.shape}")
        encoded = self.encoder(img)
        # print(f"1 encoded.requires_grad = {encoded.requires_grad}")

        if self.quantize == True:
            quatized =  Quantize(encoded)
        else:
            quatized = encoded

        encoded =   Awgn(quatized, snr = self.snr)
        # print(f"2 encoded.requires_grad = {encoded.requires_grad}")


        decoded = self.decoder(encoded)
        # print(f"3 decoded.requires_grad = {decoded.requires_grad}")
        return decoded

    def set_snr(self, snr):
        self.snr = snr

    def save(self, savedir, comp, snr, name = "AE_cnn_mnist"):
        save = os.path.join(savedir, f"{name}_comp={comp:.2f}_snr={snr:.0f}.pt")
        torch.save(self.model.state_dict(), save)
        return


X = torch.randint(low = 0, high= 255, size = (128, 1, 28, 28)) * 1.0
ae = AED_cnn_mnist(100, snr = None)
y = ae(X)




#打印某一层的参数名
for name in ae.state_dict():
    print(name)
#Then  I konw that the name of target layer is '1.weight'

#schemem1(recommended)
print(f"ae.state_dict()['encoder.encoder_cnn.0.weight'] = \n    {ae.state_dict()['encoder.encoder_cnn.0.weight']}")


#打印每一层的参数名和参数值
for name in ae.state_dict():
    print(f" name = {name}  \n ae.state_dict()[{name}] = {ae.state_dict()[name]}")
    # print(name)
    # print(ae.state_dict()[name])



#打印每一层的参数名和参数值
params = list(ae.named_parameters())#get the index by debuging
l = len(params)
for i in range(l):
    # print(params[i][0])              # name
    # print(params[i][1].data)         # data
    print(f" params[{i}][0] = {params[i][0]}, \n params[{i}][1].data = \n      {params[i][1].data}")



#打印每一层的参数名和参数值
params = {}#change the tpye of 'generator' into dict
for name, param in ae.named_parameters():
    params[name] = param.detach().cpu().numpy()
    print(f"name = {name}, params[{name}] = \n{params[name]}")



#打印每一层的参数名和参数值
#schemem1(recommended)
for name, param in ae.named_parameters():
    # print(f"  name = {name}\n  param = \n    {param}")
    print(f"  name = {name}\n  param.data = \n    {param.data}")



#scheme4
for layer in ae.modules():
    if(isinstance(layer, nn.Conv2d)):
        print(layer.weight)





#===================================================================================
# 测试在init和forward部分，模型的层的定义和调用对模型结构的关系
#===================================================================================

# 定义一个简单的网络
class net(nn.Module):
    def __init__(self, num_class=10):
        super(net, self).__init__()
        self.fc1 = nn.Linear(8, 4)
        self.fc2 = nn.Linear(4, num_class)


    def forward(self, x):
        return self.fc2(self.fc1(x))

model = net()

print(f"模型结构为：\n{model}, \n模型参数为:\n ")
for name, param in  model.named_parameters():
    if param.requires_grad:
        #print(f"{name}: {param.size()}, {param.requires_grad} ")
        print(f"{name: <25}: size={param.size()}, requires_grad={param.requires_grad} ")

# 模型结构为：
# net(
#   (fc1): Linear(in_features=8, out_features=4, bias=True)
#   (fc2): Linear(in_features=4, out_features=10, bias=True)
# ),
# 模型参数为:

# fc1.weight               : size=torch.Size([4, 8]), requires_grad=True
# fc1.bias                 : size=torch.Size([4]), requires_grad=True
# fc2.weight               : size=torch.Size([10, 4]), requires_grad=True
# fc2.bias                 : size=torch.Size([10]), requires_grad=True


#==================================================================================
# 定义一个简单的网络
class net1(nn.Module):
    def __init__(self, num_class=10):
        super(net1, self).__init__()
        self.fc2 = nn.Linear(4, num_class)
        self.fc1 = nn.Linear(8, 4)


    def forward(self, x):
        return self.fc2(self.fc1(x))


model = net1()

print(f"模型结构为：\n{model}, \n模型参数为:\n ")
for name, param in  model.named_parameters():
    if param.requires_grad:
        #print(f"{name}: {param.size()}, {param.requires_grad} ")
        print(f"{name: <25}: size={param.size()}, requires_grad={param.requires_grad} ")

# 模型结构为：
# net(
#   (fc2): Linear(in_features=4, out_features=10, bias=True)
#   (fc1): Linear(in_features=8, out_features=4, bias=True)
# ),
# 模型参数为:

# fc2.weight               : size=torch.Size([10, 4]), requires_grad=True
# fc2.bias                 : size=torch.Size([10]), requires_grad=True
# fc1.weight               : size=torch.Size([4, 8]), requires_grad=True
# fc1.bias                 : size=torch.Size([4]), requires_grad=True

#==================================================================================
# 定义一个简单的网络
class net1(nn.Module):
    def __init__(self, num_class=10):
        super(net1, self).__init__()
        self.fc2 = nn.Linear(4, num_class)
        self.fc1 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc2(self.fc1(x))


model = net1()

print(f"模型结构为：\n{model}, \n模型参数为:\n ")
for name, param in  model.named_parameters():
    if param.requires_grad:
        #print(f"{name}: {param.size()}, {param.requires_grad} ")
        print(f"{name: <25}: size={param.size()}, requires_grad={param.requires_grad} ")

# 模型结构为：
# net1(
#   (fc2): Linear(in_features=4, out_features=10, bias=True)
#   (fc1): Linear(in_features=8, out_features=4, bias=True)
#   (fc3): Linear(in_features=8, out_features=4, bias=True)
# ),
# 模型参数为:

# fc2.weight               : size=torch.Size([10, 4]), requires_grad=True
# fc2.bias                 : size=torch.Size([10]), requires_grad=True
# fc1.weight               : size=torch.Size([4, 8]), requires_grad=True
# fc1.bias                 : size=torch.Size([4]), requires_grad=True
# fc3.weight               : size=torch.Size([4, 8]), requires_grad=True
# fc3.bias                 : size=torch.Size([4]), requires_grad=True

#==================================================================================
# 定义一个简单的网络
class net1(nn.Module):
    def __init__(self, num_class=10):
        super(net1, self).__init__()
        self.fc2 = nn.Linear(4, num_class)
        self.fc1 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc2(self.fc1(x))


model = net1()

for name, param in model.named_parameters():
    if "fc1" in name:
        param.requires_grad = False


print(f"模型结构为：\n{model}, \n模型参数为:\n ")
for name, param in  model.named_parameters():
    print(f"{name: <25}: size={param.size()}, requires_grad={param.requires_grad} ")

# 模型结构为：
# net1(
#   (fc2): Linear(in_features=4, out_features=10, bias=True)
#   (fc1): Linear(in_features=8, out_features=4, bias=True)
#   (fc3): Linear(in_features=8, out_features=4, bias=True)
# ),
# 模型参数为:

# fc2.weight               : size=torch.Size([10, 4]), requires_grad=True
# fc2.bias                 : size=torch.Size([10]), requires_grad=True
# fc1.weight               : size=torch.Size([4, 8]), requires_grad=False
# fc1.bias                 : size=torch.Size([4]), requires_grad=False
# fc3.weight               : size=torch.Size([4, 8]), requires_grad=True
# fc3.bias                 : size=torch.Size([4]), requires_grad=True


#  由以上几个例子可见，模型的结构只由模型定义的顺序决定，与模型层的调用先后没关系，即使某层定义了，没被调用，也会存在于模型结构中。


#===================================================================================
## 测试保存模型，模型大小是否改变
#===================================================================================
# 定义一个简单的网络
class net(nn.Module):
    def __init__(self, num_class=10):
        super(net, self).__init__()
        self.fc1 = nn.Linear(8, 4)
        self.fc2 = nn.Linear(4, num_class)


    def forward(self, x):
        return self.fc2(self.fc1(x))


model = net()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)  # 传入的是所有的参数


for epoch in range(1000):
    x = torch.randn((3, 8))
    label = torch.randint(0,10,[3]).long()

    output = model(x)
    #print(f"epoch = {epoch}, x.shape = {x.shape}, output.shape = {output.shape}")
    loss = loss_fn(output, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f"epoch = {epoch+1}, model.fc1.weight = \n{model.fc1.weight}\n")
        #time.sleep(2)
        PATH = "/home/jack/snap/model/{}_epoch.pt".format(epoch+1)
        torch.save(model.state_dict(), PATH)


print(f"\n\n")
model1 = net()
path1 = "/home/jack/snap/model/{}_epoch.pt".format(20)
model1.load_state_dict(torch.load(path1))
print(f"model1.fc1.weight = {model1.fc1.weight}", )
#print(f"model1.fc2.weight = {model1.fc2.weight}\n",)



model2 = net()
path2 = "/home/jack/snap/model/{}_epoch.pt".format(100)
model2.load_state_dict(torch.load(path2))
print(f"model2.fc1.weight = {model2.fc1.weight}", )
#print(f"model2.fc2.weight = {model2.fc2.weight}\n",)




#===================================================================================
# 测试训练过程参数是否改变以及怎么冻结参数
#===================================================================================

#===================================================================================
# 情况一：不冻结参数时
#===================================================================================
model = net()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)  # 传入的是所有的参数

# 训练前的模型参数
print(f"model.fc1.weight = {model.fc1.weight}", )
print(f"model.fc2.weight = {model.fc2.weight}\n", )



for epoch in range(1000):
    x = torch.randn((3, 8))
    label = torch.randint(0,10,[3]).long()

    output = model(x)
    # print(f"epoch = {epoch}, x.shape = {x.shape}, output.shape = {output.shape}")
    loss = loss_fn(output, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



# 训练后的模型参数
print(f"model.fc1.weight = {model.fc1.weight}", )
print(f"model.fc2.weight = {model.fc2.weight}\n",)




#===================================================================================
# 情况二：采用方式一冻结fc1层时
# 方式一
#===================================================================================


model = net()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)  # 传入的是所有的参数

# 训练前的模型参数
print(f"model.fc1.weight = {model.fc1.weight}", )
print(f"model.fc2.weight = {model.fc2.weight}\n", )


for name, param in model.named_parameters():
    if "fc1" in name:
        param.requires_grad = False



for epoch in range(1000):
    x = torch.randn((3, 8))
    label = torch.randint(0,10,[3]).long()

    output = model(x)
    # print(f"epoch = {epoch}, x.shape = {x.shape}, output.shape = {output.shape}")
    loss = loss_fn(output, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if (epoch+1) % 100 == 0:
        #print(f"epoch = {epoch+1}, model.fc1.weight = \n{model.fc1.weight}\n")
        #time.sleep(2)
        PATH = "/home/jack/snap/model/{}_epoch.pt".format(epoch+1)
        torch.save(model.state_dict(), PATH)


print(f"\n\n")
model1 = net()
path1 = "/home/jack/snap/model/{}_epoch.pt".format(100)
model1.load_state_dict(torch.load(path1))
print(f"model1.fc1.weight = {model1.fc1.weight}", )
print(f"model1.fc2.weight = {model1.fc2.weight}\n\n", )
#print(f"model1.fc2.weight = {model1.fc2.weight}\n",)



model2 = net()
path2 = "/home/jack/snap/model/{}_epoch.pt".format(1000)
model2.load_state_dict(torch.load(path2))
print(f"model2.fc1.weight = {model2.fc1.weight}", )
print(f"model2.fc2.weight = {model2.fc2.weight}", )
#print(f"model2.fc2.weight = {model2.fc2.weight}\n",)



#===================================================================================
# 情况二：采用方式一冻结fc1层时
# 方式一
#===================================================================================


# 定义一个简单的网络
class net(nn.Module):
    def __init__(self, num_class=10):
        super(net, self).__init__()
        self.fc1 = nn.Linear(8, 4)
        self.fc2 = nn.Linear(4, num_class)


    def forward(self, x):
        return self.fc2(self.fc1(x))


model = net()

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)  # 传入的是所有的参数

# 训练前的模型参数
print(f"model.fc1.weight = {model.fc1.weight}", )
print(f"model.fc2.weight = {model.fc2.weight}\n", )




model.train()
torch.set_grad_enabled(True)

for name, param in model.named_parameters():
    if "fc1" in name:
        param.requires_grad = False

for epoch in range(1000):
    x = torch.randn((3, 8))
    label = torch.randint(0,10,[3]).long()

    output = model(x)
    # print(f"epoch = {epoch}, x.shape = {x.shape}, output.shape = {output.shape}")
    loss = loss_fn(output, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if (epoch+1) % 100 == 0:
        #print(f"epoch = {epoch+1}, model.fc1.weight = \n{model.fc1.weight}\n")
        #time.sleep(2)
        PATH = "/home/jack/snap/model/{}_epoch.pt".format(epoch+1)
        torch.save(model.state_dict(), PATH)


print(f"\n\n")
model1 = net()
path1 = "/home/jack/snap/model/{}_epoch.pt".format(100)
model1.load_state_dict(torch.load(path1))
print(f"model1.fc1.weight = {model1.fc1.weight}", )
print(f"model2.fc2.weight = {model1.fc2.weight}\n\n", )
#print(f"model1.fc2.weight = {model1.fc2.weight}\n",)



model2 = net()
path2 = "/home/jack/snap/model/{}_epoch.pt".format(1000)
model2.load_state_dict(torch.load(path2))
print(f"model2.fc1.weight = {model2.fc1.weight}", )
print(f"model2.fc2.weight = {model2.fc2.weight}", )
#print(f"model2.fc2.weight = {model2.fc2.weight}\n",)



#===================================================================================
# 情况三：采用方式二冻结fc1层时
# 方式二
#===================================================================================

model = net()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc2.parameters(), lr=1e-2)  # 优化器只传入fc2的参数


# 训练前的模型参数
print(f"model.fc1.weight = {model.fc1.weight}", )
print(f"model.fc2.weight = {model.fc2.weight}\n", )

for epoch in range(1000):
    x = torch.randn((3, 8))
    label = torch.randint(0,3,[3]).long()
    output = model(x)

    loss = loss_fn(output, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if (epoch+1) % 100 == 0:
        #print(f"epoch = {epoch+1}, model.fc1.weight = \n{model.fc1.weight}\n")
        #time.sleep(2)
        PATH = "/home/jack/snap/model/{}_epoch.pt".format(epoch+1)
        torch.save(model.state_dict(), PATH)


print(f"\n\n")
model1 = net()
path1 = "/home/jack/snap/model/{}_epoch.pt".format(100)
model1.load_state_dict(torch.load(path1))
print(f"model1.fc1.weight = {model1.fc1.weight}", )
print(f"model2.fc1.weight = {model2.fc1.weight}\n\n", )
#print(f"model1.fc2.weight = {model1.fc2.weight}\n",)



model2 = net()
path2 = "/home/jack/snap/model/{}_epoch.pt".format(1000)
model2.load_state_dict(torch.load(path2))
print(f"model2.fc1.weight = {model2.fc1.weight}", )
print(f"model2.fc2.weight = {model2.fc2.weight}", )
#print(f"model2.fc2.weight = {model2.fc2.weight}\n",)

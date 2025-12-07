


import torch.nn as nn
import torch.nn.functional as func


## classifier
class LeNet_3(nn.Module):
    def __init__(self, ):
        super(LeNet_3, self).__init__()
        # input shape: 1 * 1 * 28 * 28
        self.conv = nn.Sequential(
            ## conv layer 1
            ## conv: 1, 28, 28 -> 10, 24, 24
            nn.Conv2d(1, 10, kernel_size = 5),
            ## 10, 24, 24 -> 10, 12, 12
            nn.MaxPool2d(kernel_size = 2, ),
            nn.ReLU(),

            ## 10, 12, 12 -> 20, 8, 8
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(),
            ## 20, 8, 8 -> 20, 4, 4
            nn.MaxPool2d(kernel_size = 2, ),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            ## full connect layer 1
            nn.Linear(320, 50), nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10),
            nn.LogSoftmax(dim = 1)
        )

    def forward(self, img):
        feature = self.conv(img)

        # aver = torch.mean(feature, )
        # print(f"3 aver = {aver}")
        # snr = 0.2
        # aver_noise = aver * (1 / 10 **(snr/10))
        # # print(f"{aver}, {aver_noise}")
        # noise = torch.randn(size = feature.shape) * aver_noise.to('cpu')
        # feature = feature + noise.to(feature.device)

        output = self.fc(feature.view(img.shape[0], -1))
        return output

#=============================================================================================================
#  AE based on cnn for MNIST
#=============================================================================================================
import os
import torch
from torch import nn

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
            nn.Linear(128, encoded_space_dim),
            nn.Tanh()
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

## AutoEncoder
class AED_MNIST(nn.Module):
    def __init__(self, encoded_space_dim = 100, snr  = 3, ):
        super(AED_MNIST, self).__init__()
        self.snr = snr

        self.encoder = Encoder_cnn_mnist(encoded_space_dim)
        self.decoder = Decoder_cnn_mnist(encoded_space_dim)

    def forward(self, img, attack_vector = "" ):
        # print(f"img.shape = {img.shape}")
        encoded = self.encoder(img)
        # print(f"1 encoded.requires_grad = {encoded.requires_grad}")
        # print(f"0:    {encoded.min()}, {encoded.max()}, {encoded.mean()}")

        Y =  Awgn(encoded, snr = self.snr)
        # print(f"2 encoded.requires_grad = {encoded.requires_grad}")

        decoded = self.decoder(Y)
        # print(f"3 decoded.requires_grad = {decoded.requires_grad}")
        return decoded

    def set_snr(self, snr):
        self.snr = snr

    def save(self, savedir, comp, snr, name = "AE_cnn_mnist"):
        save = os.path.join(savedir, f"{name}_comp={comp:.2f}_snr={snr:.0f}.pt")
        torch.save(self.model.state_dict(), save)
        return





















































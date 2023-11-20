# 2021.05.07-Changed for IPT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>



import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal



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
        self.encoder_lin1 = nn.Linear(3 * 3 * 32, 128)
        self.encoder_lin2 = nn.ReLU(True)
        self.encoder_lin3 = nn.Linear(128, encoded_space_dim)



    def forward(self, x):
        # print(f"1 x.shape = {x.shape}")

        x = self.encoder_cnn(x)
        # print(f"2 x.shape = {x.shape}")

        x = self.flatten(x)
        # print(f"3 x.shape = {x.shape}")

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

        # print(f"0:    {encoded.min()}, {encoded.max()}")

        if self.quantize == True:
            encoded = common.Quantize(encoded)
        else:
            pass   # quatized = encoded

        Y =  common.Awgn(encoded, snr = self.snr)
        # print(f"2 encoded.requires_grad = {encoded.requires_grad}")


        decoded = self.decoder(Y)
        # print(f"3 decoded.requires_grad = {decoded.requires_grad}")
        return decoded







































































































































































































































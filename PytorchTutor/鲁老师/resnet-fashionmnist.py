import torch
from torch import nn
from torch.nn import functional as F
import argparse
import sys
sys.path.append("..") 
import mlutils.pytorch as mlutils

class Residual(nn.Module):
    """The Residual block of ResNet."""
    def __init__(self, in_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        # 1×1 convolutional layer can change output channel number
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def resnet_block(in_channels, num_channels, num_residuals, first_block=False):
    """return multiple resnet blocks"""
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

def resnet18():
    """return resnet"""
    # convolutional and max pooling layer 
    b1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    # resnet block
    b2 = nn.Sequential(*resnet_block(in_channels=64, num_channels=64, num_residuals=2, first_block=True))
    b3 = nn.Sequential(*resnet_block(in_channels=64, num_channels=128, num_residuals=2))
    b4 = nn.Sequential(*resnet_block(in_channels=128, num_channels=256, num_residuals=2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))

    net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))

    return net

def main(args):
    net = resnet18()
    print(f"net = \n{net}\n")
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss = torch.nn.CrossEntropyLoss()
    # load data
    train_iter, test_iter = mlutils.load_data_fashion_mnist(batch_size=args.batch_size, resize=96)
    # X.shape = torch.Size([256, 1, 96, 96]), y.shape = torch.Size([256]).y_hat.shape = torch.Size([256, 10])
    # train
    mlutils.train(net, train_iter, test_iter, args.batch_size, optimizer, loss, args.num_epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image classification')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of train epochs')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    args = parser.parse_args()
    main(args)

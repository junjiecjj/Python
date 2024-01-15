import torch
from torch import nn
import argparse
import sys
sys.path.append("..") 
import mlutils.pytorch as mlutils

def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())

def NIN():
    '''
    Returns the NiN network
    '''
    # Fashion-MNIST 1 * 28 * 28, resize into the input into 1 * 224 * 224
    # input shape: 1 * 224 * 224
    net = nn.Sequential(
        # 1 * 224 * 224 -> 96 * 54 * 54
        nin_block(1, 96, kernel_size=11, strides=4, padding=0),
        # 96 * 54 * 54 -> 96 * 26 * 26
        nn.MaxPool2d(kernel_size=3, stride=2),
        # 96 * 26 * 26 -> 256 * 26 * 26
        nin_block(96, 256, kernel_size=5, strides=1, padding=2),
        # 256 * 26 * 26 -> 256 * 12 * 12
        nn.MaxPool2d(3, stride=2),
        # 256 * 12 * 12 -> 384 * 12 * 12
        nin_block(256, 384, kernel_size=3, strides=1, padding=1),
        # 384 * 12 * 12 -> 384 * 5 * 5
        nn.MaxPool2d(3, stride=2),
        nn.Dropout(0.5),
        # 384 * 5 * 5 -> 10 * 5 * 5
        nin_block(384, 10, kernel_size=3, strides=1, padding=1),
        # 10 * 5 * 5 -> 10 * 1 * 1
        nn.AdaptiveAvgPool2d((1, 1)),
        # get the final classification result
        nn.Flatten())

    return net

def main(args):
    net = NIN()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss = torch.nn.CrossEntropyLoss()
    # load data
    train_iter, test_iter = mlutils.load_data_fashion_mnist(batch_size=args.batch_size, resize=224)
    # X.shape = torch.Size([128, 1, 224, 224]), y.shape = torch.Size([128]).y_hat.shape = torch.Size([128, 10])
    # train
    mlutils.train(net, train_iter, test_iter, args.batch_size, optimizer, loss, args.num_epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image classification')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of train epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    args = parser.parse_args()
    main(args)
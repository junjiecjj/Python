import torch.nn as nn
import torch.nn.functional as F
import torch


class Net(nn.Module):
    def __init__(self,n_input,n_hidden1,n_hidden2,n_hidden3,n_output):
        super(Net,self).__init__()
        self.hidden1 = nn.Linear(n_input,n_hidden1)
        self.hidden2 = nn.Linear(n_hidden1,n_hidden2)
        self.hidden3 = nn.Linear(n_hidden2, n_hidden3)
        self.predict = nn.Linear(n_hidden3,n_output)
    def forward(self,input):
        out = self.hidden1(input)
        out = F.sigmoid(out)
        out = self.hidden2(out)
        out = F.sigmoid(out)
        out = self.hidden3(out)
        out = F.relu(out)
        out =self.predict(out)
        return out

if __name__ == '__main__':
    net = Net(8,16,8,16,8)
    input = torch.ones((100,8))
    output = net(input)
    print(output.shape)

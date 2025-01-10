


import torch
import torch.nn as nn
import torch.nn.functional as F


class Mnist_2NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, inputs):
        tensor = F.relu(self.fc1(inputs))
        tensor = F.relu(self.fc2(tensor))
        tensor = self.fc3(tensor)
        return tensor

# # Data volume = 199210 (floating point number)
# net = Mnist_2NN()
# data_valum = 0
# for key, var in net.state_dict().items():
#     data_valum += var.numel()
# print(f"Data volume = {data_valum} (floating point number) ")

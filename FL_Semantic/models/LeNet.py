


import torch.nn as nn
import torch.nn.functional as func


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = func.max_pool2d(x, 2)
        x = func.relu(self.conv2(x))
        x = func.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x



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





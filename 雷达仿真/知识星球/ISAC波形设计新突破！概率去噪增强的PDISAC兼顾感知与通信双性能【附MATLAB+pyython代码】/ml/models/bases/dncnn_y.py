import torch
import torch.nn as nn
import torch.nn.init as init

# from models.utils import weights_init_kaiming


class DnCNN(nn.Module):
    def __init__(self, depth=20, n_filters=64, kernel_size=3, img_channels=3):
        """Pytorch implementation of DnCNN.
        Parameters
        ----------
        depth : int
            Number of fully convolutional layers in dncnn. In the original paper, the authors have used depth=17 for non-
            blind denoising and depth=20 for blind denoising.
        n_filters : int
            Number of filters on each convolutional layer.
        kernel_size : int tuple
            2D Tuple specifying the size of the kernel window used to compute activations.
        n_channels : int
            Number of image channels that the network processes (1 for grayscale, 3 for RGB)
        Example
        -------
        >>> from OpenDenoising.model.architectures.pytorch import DnCNN
        >>> dncnn_s = DnCNN(depth=17)
        >>> dncnn_b = DnCNN(depth=20)
        """
        super(DnCNN, self).__init__()
        layers = [
            nn.Conv2d(
                in_channels=img_channels,
                out_channels=n_filters,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(
                    in_channels=n_filters,
                    out_channels=n_filters,
                    kernel_size=kernel_size,
                    padding=1,
                    bias=False,
                )
            )
            # layers.append(nn.BatchNorm2d(n_filters))
            layers.append(nn.ReLU(inplace=True))

        # self.lastconv = nn.Conv2d(in_channels=n_filters, out_channels=img_channels, kernel_size=kernel_size,
        #                         padding=1, bias=False)

        layers.append(
            nn.Conv2d(
                in_channels=n_filters,
                out_channels=img_channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            ),
            # nn.Tanh(inplace=True),
        )
        self.dncnn = nn.Sequential(*layers)

        # self.weight_init()

        self._initialize_weights()

    def input_mask(self, image, prob=0.5):
        """
        Multiplicative bernoulli
        """
        b, c, x, y = image.size()
        mask = torch.ones(x, y)
        mask = torch.bernoulli(mask * prob).cuda()
        mask = mask.squeeze(0).squeeze(1)
        noise_image = image * mask
        # noise_image = noise_image - value + value * mask
        return noise_image

    def forward(self, Y):
        noise = self.dncnn(Y)
        output = Y - noise

        return output

    def predict(self, Y):
        return self.forward(Y)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                if m.bias is not None:
                    init.constant_(m.bias, 0)

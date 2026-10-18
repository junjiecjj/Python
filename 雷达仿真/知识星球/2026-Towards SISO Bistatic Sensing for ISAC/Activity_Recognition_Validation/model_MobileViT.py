import torch
import torch.nn as nn
import timm


class MicroDopplerMobileViT(nn.Module):
    def __init__(self, num_classes=6, in_channels=1, backbone_name='mobilevit_xxs'):
        """
        MobileViT-based micro-Doppler classification model.

        Args:
            num_classes (int): Number of output gesture classes.
            in_channels (int): Number of input channels (e.g., 1 for micro-Doppler spectrogram).
            backbone_name (str): Name of the MobileViT backbone in timm.
        """
        super().__init__()

        # Create the MobileViT backbone (without classification head)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,  # Load pretrained weights from timm
            in_chans=in_channels,  # Number of input channels (e.g., 1 for grayscale micro-Doppler)
            features_only=False,  # Output the final feature vector, not intermediate features
            num_classes=0  # Set to 0 to remove timm's built-in classifier head
        )

        # Get the output feature dimension of the backbone (e.g., 640 for mobilevit_xxs)
        backbone_out_dim = self.backbone.num_features

        # Define the classification head
        self.classifier = nn.Linear(backbone_out_dim, num_classes)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input tensor of shape (B, in_channels, H, W)

        Returns:
            Tensor: Output logits of shape (B, num_classes)
        """
        feat = self.backbone(x)  # → Shape: (B, D), where D = backbone_out_dim
        return self.classifier(feat)  # → Shape: (B, num_classes)

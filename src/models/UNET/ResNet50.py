import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Encoder(nn.Module):
    def __init__(self, freeze=True):
        super(ResNet50Encoder, self).__init__()
        resnet50_model = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Freeze all parameters if specified
        if freeze:
            for param in resnet50_model.parameters():
                param.requires_grad = False

        # Extract the different layers
        self.conv1 = nn.Sequential(
            resnet50_model.conv1,
            resnet50_model.bn1,
            resnet50_model.relu
        )
        self.maxpool = resnet50_model.maxpool
        self.layer1 = resnet50_model.layer1  # Output: 256 channels
        self.layer2 = resnet50_model.layer2  # Output: 512 channels
        self.layer3 = resnet50_model.layer3  # Output: 1024 channels
        self.layer4 = resnet50_model.layer4  # Output: 2048 channels

    def forward(self, x):
        x1 = self.conv1(x)       # [B, 64, H/2, W/2]
        x2 = self.maxpool(x1)    # [B, 64, H/4, W/4]
        x3 = self.layer1(x2)     # [B, 256, H/4, W/4]
        x4 = self.layer2(x3)     # [B, 512, H/8, W/8]
        x5 = self.layer3(x4)     # [B, 1024, H/16, W/16]
        x6 = self.layer4(x5)     # [B, 2048, H/32, W/32]
        return x1, x3, x4, x5, x6


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle dimension mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class FinalUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class UNetResNet50(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, freeze=True):
        super(UNetResNet50, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder
        self.encoder = ResNet50Encoder(freeze=freeze)

        # Decoder - adjusted for ResNet50's larger channel dimensions
        self.up1 = Up(2048, 1024)  # 2048 -> 1024
        self.up2 = Up(1024, 512)   # 1024 -> 512
        self.up3 = Up(512, 256)    # 512 -> 256
        self.up4 = Up(256, 64)     # 256 -> 64 (skip connection from conv1 has 64 channels)
        self.final_up = FinalUp(64, 32)  # 64 -> 32

        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1, x3, x4, x5, x6 = self.encoder(x)  # x1:64, x3:256, x4:512, x5:1024, x6:2048

        # Decoder
        x = self.up1(x6, x5)   # 2048 + 1024 -> 1024
        x = self.up2(x, x4)    # 1024 + 512 -> 512
        x = self.up3(x, x3)    # 512 + 256 -> 256
        x = self.up4(x, x1)    # 256 + 64 -> 64
        x = self.final_up(x)   # 64 -> 32
        return self.outc(x)
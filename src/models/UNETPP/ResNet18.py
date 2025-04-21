import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNet18Encoder(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        if freeze:
            for param in resnet.parameters():
                param.requires_grad = False

        self.initial = nn.Sequential(
            resnet.conv1,  # 64 channels, stride 2
            resnet.bn1,
            resnet.relu,
            resnet.maxpool  # further downsample to 64x64 from 128x128
        )
        self.encoder1 = resnet.layer1  # 64 channels
        self.encoder2 = resnet.layer2  # 128 channels
        self.encoder3 = resnet.layer3  # 256 channels
        self.encoder4 = resnet.layer4  # 512 channels

    def forward(self, x):
        x0 = self.initial(x)     # 64 channels
        x1 = self.encoder1(x0)   # 64 channels
        x2 = self.encoder2(x1)   # 128 channels
        x3 = self.encoder3(x2)   # 256 channels
        x4 = self.encoder4(x3)   # 512 channels
        return x0, x1, x2, x3, x4


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetPlusPlusResNet18(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, freeze=True):
        super().__init__()
        self.encoder = ResNet18Encoder(freeze=freeze)

        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 64, 32)

        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.encoder(x)
        d4 = self.up4(x4, x3)
        d3 = self.up3(d4, x2)
        d2 = self.up2(d3, x1)
        d1 = self.up1(d2, x0)
        out = self.final(d1)

        # Upsample final output to match input size
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        return out

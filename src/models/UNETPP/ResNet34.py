import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights

class ResNet34Encoder(nn.Module):
    def __init__(self, freeze=True):
        super(ResNet34Encoder, self).__init__()
        resnet = resnet34(weights=ResNet34_Weights.DEFAULT)
        
        if freeze:
            for param in resnet.parameters():
                param.requires_grad = False

        self.conv1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 64
        self.layer2 = resnet.layer2  # 128
        self.layer3 = resnet.layer3  # 256
        self.layer4 = resnet.layer4  # 512

    def forward(self, x):
        x0 = self.conv1(x)          # [B, 64, H/2, W/2]
        x1 = self.maxpool(x0)       # [B, 64, H/4, W/4]
        x1 = self.layer1(x1)        # [B, 64, H/4, W/4]
        x2 = self.layer2(x1)        # [B, 128, H/8, W/8]
        x3 = self.layer3(x2)        # [B, 256, H/16, W/16]
        x4 = self.layer4(x3)        # [B, 512, H/32, W/32]
        return [x0, x1, x2, x3, x4]

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
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

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNetPlusPlusResNet34(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, freeze=True):
        super(UNetPlusPlusResNet34, self).__init__()
        self.encoder = ResNet34Encoder(freeze=freeze)
        
        self.up1 = DecoderBlock(512, 256)
        self.up2 = DecoderBlock(256, 128)
        self.up3 = DecoderBlock(128, 64)
        self.up4 = DecoderBlock(64, 64)

        self.final = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, kernel_size=1)
        )

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.encoder(x)

        d1 = self.up1(x4, x3)
        d2 = self.up2(d1, x2)
        d3 = self.up3(d2, x1)
        d4 = self.up4(d3, x0)

        output = self.final(d4)
        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode='bilinear', align_corners=True)

        return output
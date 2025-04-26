import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

class VGG19Encoder(nn.Module):
    def __init__(self, freeze=True):
        super(VGG19Encoder, self).__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT)

        if freeze:
            for param in vgg.parameters():
                param.requires_grad = False

        features = vgg.features

        self.enc1 = features[:5]     # conv1_1 to relu1_2 (64)
        self.enc2 = features[5:10]   # conv2_1 to relu2_2 (128)
        self.enc3 = features[10:19]  # conv3_1 to relu3_4 (256)
        self.enc4 = features[19:28]  # conv4_1 to relu4_4 (512)
        self.enc5 = features[28:37]  # conv5_1 to relu5_4 (512)

    def forward(self, x):
        x1 = self.enc1(x)  # [B, 64, H/2, W/2]
        x2 = self.enc2(x1) # [B, 128, H/4, W/4]
        x3 = self.enc3(x2) # [B, 256, H/8, W/8]
        x4 = self.enc4(x3) # [B, 512, H/16, W/16]
        x5 = self.enc5(x4) # [B, 512, H/32, W/32]
        return [x1, x2, x3, x4, x5]

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ResUNetVGG19(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, freeze=True):
        super(ResUNetVGG19, self).__init__()
        self.encoder = VGG19Encoder(freeze=freeze)

        self.up1 = DecoderBlock(512, 512)
        self.up2 = DecoderBlock(512, 256)
        self.up3 = DecoderBlock(256, 128)
        self.up4 = DecoderBlock(128, 64)

        self.final = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, kernel_size=1)
        )

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)

        d1 = self.up1(x5, x4)  # 512 + 512
        d2 = self.up2(d1, x3)  # 512 + 256
        d3 = self.up3(d2, x2)  # 256 + 128
        d4 = self.up4(d3, x1)  # 128 + 64

        output = self.final(d4)
        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode='bilinear', align_corners=True)

        return output
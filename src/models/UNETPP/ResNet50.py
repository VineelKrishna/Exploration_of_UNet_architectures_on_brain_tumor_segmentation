import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50Encoder(nn.Module):
    def __init__(self, freeze=True):
        super(ResNet50Encoder, self).__init__()
        resnet50_model = resnet50(weights=ResNet50_Weights.DEFAULT)

        if freeze:
            for param in resnet50_model.parameters():
                param.requires_grad = False

        self.conv1 = nn.Sequential(
            resnet50_model.conv1,
            resnet50_model.bn1,
            resnet50_model.relu
        )
        self.maxpool = resnet50_model.maxpool
        self.layer1 = resnet50_model.layer1
        self.layer2 = resnet50_model.layer2
        self.layer3 = resnet50_model.layer3
        self.layer4 = resnet50_model.layer4

    def forward(self, x):
        x0 = self.conv1(x)          # [B, 64, H/2, W/2]
        x1 = self.maxpool(x0)       # [B, 64, H/4, W/4]
        x1 = self.layer1(x1)        # [B, 256, H/4, W/4]
        x2 = self.layer2(x1)        # [B, 512, H/8, W/8]
        x3 = self.layer3(x2)        # [B, 1024, H/16, W/16]
        x4 = self.layer4(x3)        # [B, 2048, H/32, W/32]
        return [x0, x1, x2, x3, x4]

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)  # Multiply by 2 for skip connection

    def forward(self, x, skip):
        x = self.up(x)
        # Handle potential size mismatches with interpolation
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

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

class UNetPlusPlusResNet50(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, freeze=True):
        super(UNetPlusPlusResNet50, self).__init__()
        self.encoder = ResNet50Encoder(freeze=freeze)
        
        # Decoder blocks
        self.up1 = DecoderBlock(2048, 1024)  # x4 (2048) -> 1024
        self.up2 = DecoderBlock(1024, 512)   # x3 (1024) -> 512
        self.up3 = DecoderBlock(512, 256)    # x2 (512) -> 256
        self.up4 = DecoderBlock(256, 64)     # x1 (256) -> 64
        
        # Final convolution
        self.final = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, kernel_size=1)
        )
        
    def forward(self, x):
        # Get encoder features
        features = self.encoder(x)
        x0, x1, x2, x3, x4 = features
        
        # Decoder path with skip connections
        d1 = self.up1(x4, x3)  # 2048 -> 1024 (skip from x3)
        d2 = self.up2(d1, x2)  # 1024 -> 512 (skip from x2)
        d3 = self.up3(d2, x1)  # 512 -> 256 (skip from x1)
        d4 = self.up4(d3, x0)  # 256 -> 64 (skip from x0)
        
        # Final upsampling to original size
        output = self.final(d4)
        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode='bilinear', align_corners=True)
        
        return output
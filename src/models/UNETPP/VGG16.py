import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn, VGG16_BN_Weights

class VGG16BNEncoder(nn.Module):
    def __init__(self, freeze=True):
        super(VGG16BNEncoder, self).__init__()
        vgg16_model = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT).features
        
        if freeze:
            for param in vgg16_model.parameters():
                param.requires_grad = False

        # VGG16-BN has 5 blocks with maxpool after each
        self.block1 = nn.Sequential(*vgg16_model[:6])       # 64 channels
        self.block2 = nn.Sequential(*vgg16_model[6:13])     # 128 channels
        self.block3 = nn.Sequential(*vgg16_model[13:23])    # 256 channels
        self.block4 = nn.Sequential(*vgg16_model[23:33])    # 512 channels
        self.block5 = nn.Sequential(*vgg16_model[33:43])    # 512 channels

    def forward(self, x):
        x1 = self.block1(x)      # [B, 64, H/2, W/2]
        x2 = self.block2(x1)     # [B, 128, H/4, W/4]
        x3 = self.block3(x2)     # [B, 256, H/8, W/8]
        x4 = self.block4(x3)     # [B, 512, H/16, W/16]
        x5 = self.block5(x4)     # [B, 512, H/32, W/32]
        return [x1, x2, x3, x4, x5]

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

class UNetPlusPlusVGG16BN(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, freeze=True):
        super(UNetPlusPlusVGG16BN, self).__init__()
        self.encoder = VGG16BNEncoder(freeze=freeze)
        
        # Decoder blocks
        self.up1 = DecoderBlock(512, 512)    # x5 (512) -> 512
        self.up2 = DecoderBlock(512, 256)    # x4 (512) -> 256
        self.up3 = DecoderBlock(256, 128)    # x3 (256) -> 128
        self.up4 = DecoderBlock(128, 64)     # x2 (128) -> 64
        
        # Final convolution
        self.final = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, kernel_size=1)
        )
        
    def forward(self, x):
        # Get encoder features
        features = self.encoder(x)
        x1, x2, x3, x4, x5 = features
        
        # Decoder path with skip connections
        d1 = self.up1(x5, x4)  # 512 -> 512 (skip from x4)
        d2 = self.up2(d1, x3)  # 512 -> 256 (skip from x3)
        d3 = self.up3(d2, x2)  # 256 -> 128 (skip from x2)
        d4 = self.up4(d3, x1)  # 128 -> 64 (skip from x1)
        
        # Final upsampling to original size
        output = self.final(d4)
        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode='bilinear', align_corners=True)
        
        return output
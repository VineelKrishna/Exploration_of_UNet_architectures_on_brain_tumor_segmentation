import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights


class VGG19Encoder(nn.Module):
    def __init__(self, freeze=True):
        super(VGG19Encoder, self).__init__()
        vgg19_model = vgg19(weights=VGG19_Weights.DEFAULT).features

        # Freeze all parameters if specified
        if freeze:
            for param in vgg19_model.parameters():
                param.requires_grad = False

        # Extract layers for skip connections (VGG19 has more conv layers per block)
        self.conv1 = nn.Sequential(
            vgg19_model[0],  # conv1_1
            vgg19_model[1],  # relu
            vgg19_model[2],  # conv1_2
            vgg19_model[3],  # relu
            vgg19_model[4]   # maxpool
        )
        self.conv2 = nn.Sequential(
            vgg19_model[5],  # conv2_1
            vgg19_model[6],  # relu
            vgg19_model[7],  # conv2_2
            vgg19_model[8],  # relu
            vgg19_model[9]   # maxpool
        )
        self.conv3 = nn.Sequential(
            vgg19_model[10],  # conv3_1
            vgg19_model[11],  # relu
            vgg19_model[12],  # conv3_2
            vgg19_model[13],  # relu
            vgg19_model[14],  # conv3_3
            vgg19_model[15],  # relu
            vgg19_model[16],  # conv3_4
            vgg19_model[17],  # relu
            vgg19_model[18]   # maxpool
        )
        self.conv4 = nn.Sequential(
            vgg19_model[19],  # conv4_1
            vgg19_model[20],  # relu
            vgg19_model[21],  # conv4_2
            vgg19_model[22],  # relu
            vgg19_model[23],  # conv4_3
            vgg19_model[24],  # relu
            vgg19_model[25],  # conv4_4
            vgg19_model[26],  # relu
            vgg19_model[27]   # maxpool
        )
        self.conv5 = nn.Sequential(
            vgg19_model[28],  # conv5_1
            vgg19_model[29],  # relu
            vgg19_model[30],  # conv5_2
            vgg19_model[31],  # relu
            vgg19_model[32],  # conv5_3
            vgg19_model[33],  # relu
            vgg19_model[34],  # conv5_4
            vgg19_model[35],  # relu
            vgg19_model[36]   # maxpool
        )

    def forward(self, x):
        x1 = self.conv1(x)     # [B, 64, H/2, W/2]
        x2 = self.conv2(x1)    # [B, 128, H/4, W/4]
        x3 = self.conv3(x2)    # [B, 256, H/8, W/8]
        x4 = self.conv4(x3)    # [B, 512, H/16, W/16]
        x5 = self.conv5(x4)    # [B, 512, H/32, W/32]
        return x1, x2, x3, x4, x5


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


class UNetVGG19(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, freeze=True):
        super(UNetVGG19, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder
        self.encoder = VGG19Encoder(freeze=freeze)

        # Decoder
        self.up1 = Up(512, 512)    # 512 -> 512
        self.up2 = Up(512, 256)    # 512 -> 256
        self.up3 = Up(256, 128)    # 256 -> 128
        self.up4 = Up(128, 64)     # 128 -> 64
        self.final_up = FinalUp(64, 32)  # 64 -> 32

        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1, x2, x3, x4, x5 = self.encoder(x)  # x1:64, x2:128, x3:256, x4:512, x5:512

        # Decoder
        x = self.up1(x5, x4)   # 512 + 512 -> 512
        x = self.up2(x, x3)    # 512 + 256 -> 256
        x = self.up3(x, x2)    # 256 + 128 -> 128
        x = self.up4(x, x1)    # 128 + 64 -> 64
        x = self.final_up(x)   # 64 -> 32
        return self.outc(x)
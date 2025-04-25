import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


class VGG16Encoder(nn.Module):
    def __init__(self, freeze=True):
        super(VGG16Encoder, self).__init__()
        vgg16_model = vgg16(weights=VGG16_Weights.DEFAULT).features

        # Freeze all parameters if specified
        if freeze:
            for param in vgg16_model.parameters():
                param.requires_grad = False

        # Extract layers for skip connections
        self.conv1 = nn.Sequential(
            vgg16_model[0],  # conv1_1
            vgg16_model[1],  # relu
            vgg16_model[2],  # conv1_2
            vgg16_model[3],  # relu
            vgg16_model[4]   # maxpool
        )
        self.conv2 = nn.Sequential(
            vgg16_model[5],  # conv2_1
            vgg16_model[6],  # relu
            vgg16_model[7],  # conv2_2
            vgg16_model[8],  # relu
            vgg16_model[9]   # maxpool
        )
        self.conv3 = nn.Sequential(
            vgg16_model[10],  # conv3_1
            vgg16_model[11],  # relu
            vgg16_model[12],  # conv3_2
            vgg16_model[13],  # relu
            vgg16_model[14],  # conv3_3
            vgg16_model[15],  # relu
            vgg16_model[16]   # maxpool
        )
        self.conv4 = nn.Sequential(
            vgg16_model[17],  # conv4_1
            vgg16_model[18],  # relu
            vgg16_model[19],  # conv4_2
            vgg16_model[20],  # relu
            vgg16_model[21],  # conv4_3
            vgg16_model[22],  # relu
            vgg16_model[23]   # maxpool
        )
        self.conv5 = nn.Sequential(
            vgg16_model[24],  # conv5_1
            vgg16_model[25],  # relu
            vgg16_model[26],  # conv5_2
            vgg16_model[27],  # relu
            vgg16_model[28],  # conv5_3
            vgg16_model[29],  # relu
            vgg16_model[30]   # maxpool
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


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, attn_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attn = AttentionBlock(F_g=out_channels, F_l=out_channels, F_int=attn_channels)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle dimension mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Apply attention gate
        x2 = self.attn(x1, x2)
        
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


class AttUNetVGG16(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, freeze=True):
        super(AttUNetVGG16, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder
        self.encoder = VGG16Encoder(freeze=freeze)

        # Decoder with attention gates
        self.up1 = Up(512, 512, attn_channels=256)    # 512 -> 512
        self.up2 = Up(512, 256, attn_channels=128)    # 512 -> 256
        self.up3 = Up(256, 128, attn_channels=64)     # 256 -> 128
        self.up4 = Up(128, 64, attn_channels=32)      # 128 -> 64
        self.final_up = FinalUp(64, 32)               # 64 -> 32

        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1, x2, x3, x4, x5 = self.encoder(x)  # x1:64, x2:128, x3:256, x4:512, x5:512

        # Decoder with attention
        x = self.up1(x5, x4)   # 512 + 512 -> 512
        x = self.up2(x, x3)    # 512 + 256 -> 256
        x = self.up3(x, x2)    # 256 + 128 -> 128
        x = self.up4(x, x1)    # 128 + 64 -> 64
        x = self.final_up(x)   # 64 -> 32
        return self.outc(x)
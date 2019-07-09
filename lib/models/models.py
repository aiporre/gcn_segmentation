import torch
import torch.nn as nn

# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2]-x1.size()[2]
        diffX = x2.size()[3]-x1.size()[3]

        x1 = F.pad(x1, (diffX//2, diffX-diffX//2,
                        diffY//2, diffY-diffY//2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class UnetSim(nn.Module):

    def __init__(self):

        super(UnetSim, self).__init__()

        # Conv block 1 - Down 1
        self.conv1_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
        )
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 2 - Down 2
        self.conv2_block = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
        )
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 3 - Down 3
        self.conv3_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
        )
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 4 - Down 4
        self.conv4_block = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
        )
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 5 - Down 5
        self.conv5_block = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
        )

        # Up 1
        self.up_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)

        # Up Conv block 1
        self.conv_up_1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256,
                      kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
        )

        # Up 2
        self.up_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)

        # Up Conv block 2
        self.conv_up_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128,
                      kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
        )

        # Up 3
        self.up_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)

        # Up Conv block 3
        self.conv_up_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64,
                      kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
        )

        # Up 4
        self.up_4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)

        # Up Conv block 4
        self.conv_up_4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=3, padding=0, stride=1),
            nn.ReLU(inplace=True),
        )

        # Final output
        self.conv_final = nn.Conv2d(in_channels=32, out_channels=2,
                                    kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        # print('input', x.shape)

        # Down 1
        x = self.conv1_block(x)
        # print('after conv1', x.shape)
        conv1_out = x  # Save out1
        conv1_dim = x.shape[2]
        x = self.max1(x)
        # print('before conv2', x.shape)

        # Down 2
        x = self.conv2_block(x)
        # print('after conv2', x.shape)
        conv2_out = x
        conv2_dim = x.shape[2]
        x = self.max2(x)
        # print('before conv3', x.shape)

        # Down 3
        x = self.conv3_block(x)
        # print('after conv3', x.shape)
        conv3_out = x
        conv3_dim = x.shape[2]
        x = self.max3(x)
        # print('before conv4', x.shape)

        # Down 4
        x = self.conv4_block(x)
        # print('after conv5', x.shape)
        conv4_out = x
        conv4_dim = x.shape[2]
        x = self.max4(x)

        # Midpoint
        x = self.conv5_block(x)

        # Up 1
        x = self.up_1(x)
        # print('up_1', x.shape)
        lower = int((conv4_dim - x.shape[2]) / 2)
        upper = int(conv4_dim - lower)
        conv4_out_modified = conv4_out[:, :, lower:upper, lower:upper]
        x = torch.cat([x, conv4_out_modified], dim=1)
        # print('after cat_1', x.shape)
        x = self.conv_up_1(x)
        # print('after conv_1', x.shape)

        # Up 2
        x = self.up_2(x)
        # print('up_2', x.shape)
        lower = int((conv3_dim - x.shape[2]) / 2)
        upper = int(conv3_dim - lower)
        conv3_out_modified = conv3_out[:, :, lower:upper, lower:upper]
        x = torch.cat([x, conv3_out_modified], dim=1)
        # print('after cat_2', x.shape)
        x = self.conv_up_2(x)
        # print('after conv_2', x.shape)

        # Up 3
        x = self.up_3(x)
        # print('up_3', x.shape)
        lower = int((conv2_dim - x.shape[2]) / 2)
        upper = int(conv2_dim - lower)
        conv2_out_modified = conv2_out[:, :, lower:upper, lower:upper]
        x = torch.cat([x, conv2_out_modified], dim=1)
        # print('after cat_3', x.shape)
        x = self.conv_up_3(x)
        # print('after conv_3', x.shape)

        # Up 4
        x = self.up_4(x)
        # print('up_4', x.shape)
        lower = int((conv1_dim - x.shape[2]) / 2)
        upper = int(conv1_dim - lower)
        conv1_out_modified = conv1_out[:, :, lower:upper, lower:upper]
        x = torch.cat([x, conv1_out_modified], dim=1)
        # print('after cat_4', x.shape)
        x = self.conv_up_4(x)
        # print('after conv_4', x.shape)

        # Final output
        x = self.conv_final(x)

        return x

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return torch.sigmoid(x)

class FCN(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(FCN, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)


if __name__ == "__main__":
    # A full forward pass
    im = torch.randn(1, 1, 64, 84)
    # model = UnetSim()
    model = UNet(n_channels=1,n_classes=2)
    x = model(im)
    print(x.shape)
    del model
    del x
    # print(x.shape)
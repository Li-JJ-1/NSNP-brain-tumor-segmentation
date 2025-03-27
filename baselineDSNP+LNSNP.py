import torch
import torch.nn as nn



class Dilated_Conv3D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=2, dilation=2):
        super(Dilated_Conv3D, self).__init__()
        self.Conv1 = nn.Sequential(
            # nn.GroupNorm(in_ch // 2, in_ch),
            nn.BatchNorm3d(in_ch),
            nn.LeakyReLU(),
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, dilation=dilation),
            # nn.BatchNorm3d(out_ch),
            # nn.ReLU()
        )

    def forward(self, x):
        x1 = self.Conv1(x)
        return x1

class ConvSNP(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(ConvSNP,self).__init__()
        self.ConvSNP_1 = nn.Sequential(
            nn.BatchNorm3d(in_ch),
            nn.LeakyReLU(),
            nn.Conv3d(in_ch,out_ch,kernel_size=1),
        )
    def forward(self,x):
        x = self.ConvSNP_1(x)
        return x



class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        # x = self.sigmoid(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv_up = nn.Conv3d(in_ch, in_ch // 2, kernel_size=1)
        self.conv1 = ConvSNP(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.conv_up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv1(x)
        return x


class BaselineMRF(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(BaselineMRF, self).__init__()
        features = [8, 16, 32, 64, 128]

        # self.inc = InConv(in_channels, features[0])
        # self.down1 = Down(features[0], features[1])
        # self.down2 = Down(features[1], features[2])
        # self.down3 = Down(features[2], features[3])
        self.dila2_0 = Dilated_Conv3D(in_channels, in_channels * 2, dilation=2, padding=2)
        self.dila2 = Dilated_Conv3D(in_channels * 2, in_channels * 4, dilation=2, padding=2)
        self.dila3 = Dilated_Conv3D(in_channels * 4, in_channels * 8, dilation=3, padding=3)
        self.dila5 = Dilated_Conv3D(in_channels * 8, in_channels * 16, dilation=5, padding=5)
        self.dila7 = Dilated_Conv3D(in_channels * 16, in_channels * 32, dilation=6, padding=6)

        self.maxpool = nn.MaxPool3d(2,2)

        self.up1 = Up(features[4], features[3])
        self.up2 = Up(features[3], features[2])
        self.up3 = Up(features[2], features[1])
        self.up4 = Up(features[1], features[0])
        self.outc = OutConv(features[0], num_classes)

    def forward(self, x):
        x1 = self.dila2_0(x)

        x2 = self.maxpool(x1)
        x3 = self.dila2(x2)

        x4 = self.maxpool(x3)
        x5 = self.dila3(x4)

        x6 = self.maxpool(x5)
        x7 = self.dila5(x6)

        x8 = self.maxpool(x7)
        x9 = self.dila7(x8)

        x = self.up1(x9, x7)
        x = self.up2(x, x5)
        x = self.up3(x, x3)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


if __name__ == '__main__':
    x = torch.randn(1, 4, 160, 160, 128)
    net = BaselineMRF(in_channels=4, num_classes=4)
    y = net(x)
    print("params: ", sum(p.numel() for p in net.parameters()))
    print(y.shape)

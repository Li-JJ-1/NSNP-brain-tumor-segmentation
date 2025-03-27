import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import to_2tuple, trunc_normal_,DropPath



#定义ConvSNP
class ConvSNP(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(ConvSNP,self).__init__()
        self.ConvSNP_1 = nn.Sequential(
            nn.BatchNorm3d(in_ch),
            nn.LeakyReLU(),
            nn.Conv3d(in_ch ,out_ch,kernel_size=1),
        )
    def forward(self,x):
        x = self.ConvSNP_1(x)
        return x

# 定义DSB
class DSB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DSB, self).__init__()
        self.step1 = nn.Sequential(
            nn.BatchNorm3d(in_ch),
            nn.LeakyReLU(),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, dilation=2, padding=2),#膨胀SNP卷积
        )
        self.step2 = nn.Sequential(
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(),
            nn.Conv3d(in_ch, out_ch, kernel_size=2,stride=2)#snp
        )

    def forward(self, x):
        x1 = self.step1(x)
        x2 = self.step2(x1)
        return x2


# 定义DDSP模块
class DDSP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DDSP, self).__init__()
        self.maxpool = nn.MaxPool3d(2)
        self.conv1 = DSB(in_ch, out_ch)
        self.conv2 = nn.Sequential(
            nn.BatchNorm3d(2*in_ch),
            nn.LeakyReLU(),
            nn.Conv3d(2*out_ch, out_ch, kernel_size=1),
           )
        self.conv3 = nn.Sequential(
            nn.BatchNorm3d(in_ch),
            nn.LeakyReLU(),
            nn.Conv3d(in_ch, out_ch, kernel_size=2, stride=2)
        )

    def forward(self, x):
            x1 = self.maxpool(x)
            x2 = self.conv1(x)
            x3 = self.conv3(x)
            x4 = torch.cat((x1, x2), dim=1)
            x5 = self.conv2(x4)
            x6 = torch.add(x5, x3)
            return x6



# 定义膨胀卷积
class Dilated_Conv3D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=2, padding=2, dilation=2):
        super(Dilated_Conv3D, self).__init__()
        self.Conv1 = nn.Sequential(
            nn.BatchNorm3d(in_ch),
            nn.LeakyReLU(),
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, dilation=dilation),

        )

    def forward(self, x):
        x1 = self.Conv1(x)
        return x1


class ICA_block(nn.Module):
    def __init__(self, in_ch_1, in_ch_2, in_ch,
                 drop_rate=0.1):
        super(ICA_block, self).__init__()


        self.inplanes = in_ch
        r = 4
        L = 8
        d = max(int(self.inplanes / r), L)
        self.fc1 = nn.Linear(2 * self.inplanes, d)
        self.fc2 = nn.Linear(d, self.inplanes)
        # self.fc3 = nn.Linear(d, self.inplanes)
        self.dropout = nn.Dropout(drop_rate)
        self.relu = nn.ReLU()
        # self.conv = nn.Conv3d(2 * in_ch, in_ch, kernel_size=1)
        self.conv1 = nn.Conv3d(2, 1, kernel_size=1)
        # self.conv2 = nn.Conv3d(1, 1, kernel_size=3, padding=1)
    def forward(self, x):
        """
        串行
        """
        # B C D H W  通道
        avg_pool = F.avg_pool3d(x, (x.size(2), x.size(3), x.size(4)),stride=(x.size(2), x.size(3), x.size(4)))
        max_pool = F.max_pool3d(x, (x.size(2), x.size(3), x.size(4)),stride=(x.size(2), x.size(3), x.size(4)))
        sc = torch.cat([avg_pool,max_pool],1)
        sc = torch.sigmoid(sc)
        sc = sc.reshape([-1,2*self.inplanes])

        zc1 = self.fc2(self.dropout(self.relu(self.fc1(sc))))
        zc1 = torch.sigmoid(zc1).reshape([-1, self.inplanes, 1, 1,1])

        fuse = zc1 * x

        #空间注意力
        #形式1
        avg_pool1 = torch.mean(fuse, dim=1, keepdim=True)
        max_pool1 = torch.max(fuse, dim=1, keepdim=True).values

        sc1 = torch.cat([avg_pool1, max_pool1], dim=1)
        sc1 = self.conv1(sc1)
        sc1 = torch.sigmoid(sc1)
        fuse1 = sc1 * fuse

        return fuse1

class ERF_Block(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(ERF_Block, self).__init__()

        self.step1 = nn.Sequential(
            nn.GroupNorm(in_ch // 2, in_ch),
            nn.GELU(),
            nn.Conv3d(in_ch, out_ch, kernel_size=1),
            # nn.ReLU(inplace=True),

        )

        self.step2 = nn.Sequential(
            nn.GroupNorm(in_ch // 2, in_ch),
            nn.GELU(),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),

        )

        self.step3 = nn.Sequential(
            nn.GroupNorm(in_ch // 2, in_ch),
            nn.GELU(),
            nn.Conv3d(in_ch, out_ch, kernel_size=5, padding=2),
            # nn.ReLU(inplace=True),

        )

        self.step2_3 = nn.Sequential(
            nn.GroupNorm(out_ch // 2, out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=1)
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.step1(x)
        x2 = self.step2(x)
        x3 = self.step3(x)

        x2_3 = self.step2_3(x2 + x3)

        out = x2_3 + x1
        # out = x1 + x2 + x3

        return out


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool1 = nn.AvgPool3d(kernel_size=3, stride=1, padding=1, count_include_pad=False)
        self.conv = nn.Sequential(
            nn.BatchNorm3d(in_ch),
            nn.LeakyReLU(),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.pool1(x)
        x3 = x1 + x2
        return x3


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Poolformer(nn.Module):
    def __init__(self,dim, pool_size=3, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop=0.1, drop_path=0.1,
                 use_layer_scale=True, layer_scale_init_value=1e-5):
        super(Poolformer,self).__init__()
        self.norm1 = norm_layer(dim)
        self.dim = dim
        self.token_mixer = Pooling(in_ch=dim, out_ch=dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1).reshape(-1,self.dim,1,1,1)
                * self.token_mixer(self.norm1(x)))

            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1).reshape(-1,self.dim,1,1,1)
                * self.mlp(self.norm2(x)))

        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, flag):
        super(Up, self).__init__()


        self.up = nn.Upsample(scale_factor=2, mode='trilinear')
        self.conv_up = nn.Conv3d(in_ch, in_ch//2, kernel_size=1)
        self.poolformer = Poolformer(out_ch)
        self.conv2 = nn.Sequential(
            nn.BatchNorm3d(in_ch),
            nn.LeakyReLU(),
            nn.Conv3d(in_ch, out_ch, kernel_size=1)
        )

        if flag == 'a':
            self.skip = ICA_block(out_ch, out_ch, out_ch)
        elif flag == 'b':
            self.skip = ICA_block(out_ch, out_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.conv_up(x1)
        if self.skip is not None:
            x2 = self.skip(x2)

        x = torch.cat([x2, x1], dim=1)
        # x = x1 + x2
        x = self.conv2(x)
        x = self.poolformer(x)

        return x


class MyNet(nn.Module):

    def __init__(self, in_channels, num_classes):  # inch =4
        super(MyNet, self).__init__()
        self.channel_list = [8, 16, 32, 64, 128]

        self.dila2_0 = Dilated_Conv3D(in_channels, in_channels * 2, dilation=2, padding=2)
        self.dila2 = Dilated_Conv3D(in_channels * 2, in_channels * 4, dilation=2, padding=2)
        self.dila3 = Dilated_Conv3D(in_channels * 4, in_channels * 8, dilation=3, padding=3)
        self.dila5 = Dilated_Conv3D(in_channels * 8, in_channels * 16, dilation=5, padding=5)
        self.dila7 = Dilated_Conv3D(in_channels * 16, in_channels * 32, dilation=6, padding=6)

        self.ddsp0 = DDSP(8, 8)
        self.ddsp1 = DDSP(16, 16)
        self.ddsp2 = DDSP(32, 32)
        self.ddsp3 = DDSP(64, 64)


        self.up1 = Up(self.channel_list[4], self.channel_list[3], 'b')
        self.up2 = Up(self.channel_list[3], self.channel_list[2], 'b')
        self.up3 = Up(self.channel_list[2], self.channel_list[1], 'b')
        self.up4 = Up(self.channel_list[1], self.channel_list[0], 'b')

        self.out_conv = nn.Conv3d(self.channel_list[0], num_classes, kernel_size=1)



    def forward(self, x):
        #  第一层
        x0 = self.dila2_0(x)
        # 第二层
        x1_0 = self.ddsp0(x0)
        x1 = self.dila2(x1_0)
        # 第三层
        x2 = self.ddsp1(x1)
        x3 = self.dila3(x2)
        # 第四层
        x4 = self.ddsp2(x3)
        x5 = self.dila5(x4)
        # 第五层
        x6 = self.ddsp3(x5)
        x7 = self.dila7(x6)

        x = self.up1(x7, x5)
        x = self.up2(x, x3)
        x = self.up3(x, x1)
        x = self.up4(x, x0)

        x = self.out_conv(x)

        return x



if __name__ == '__main__':

    x = torch.randn(1, 4, 128, 128, 64)  # 1代表一个样本
    net = MyNet(in_channels=4, num_classes=4)
    y = net(x)
    print("params: ", sum(p.numel() for p in net.parameters()))
    print(y.shape)

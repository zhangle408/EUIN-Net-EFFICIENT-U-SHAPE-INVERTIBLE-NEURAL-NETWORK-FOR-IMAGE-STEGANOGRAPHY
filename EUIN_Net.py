# from model import *
from invblock import INV_block_affine
import torch.nn as nn
import torch


class EUIN(nn.Module):

    def __init__(self, in_channels):
        super(EUIN, self).__init__()

        self.inv1 = INV_block_affine(in_1=3, in_2=6)
        self.down1 = Down(in_channels, in_channels * 2, pooling=True)
        self.up1 = Up(in_channels * 2, in_channels, transpose=False)
        self.conv1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

        self.inv2 = INV_block_affine(in_1=6, in_2=12)
        self.down2 = Down(in_channels * 2, in_channels * 4, pooling=True)
        self.up2 = Up(in_channels * 4, in_channels * 2, transpose=False)
        self.conv2 = nn.Conv2d(in_channels * 4, in_channels * 2, kernel_size=1)

        self.inv3 = INV_block_affine(in_1=12, in_2=24)
        self.down3 = Down(in_channels * 4, in_channels * 8, pooling=True)
        self.up3 = Up(in_channels * 8, in_channels * 4, transpose=False)
        self.conv3 = nn.Conv2d(in_channels * 8, in_channels * 4, kernel_size=1)

        self.inv4 = INV_block_affine(in_1=24, in_2=48) #[192,8,8]

        self.up5 = Up(in_channels * 8, in_channels * 4, transpose=False)  #[192,16,16] 上采样后和对面的融合，通道数又扩大二倍
        self.conv5 = nn.Conv2d(in_channels * 8, in_channels * 4, kernel_size=1)#[96,16,16] #降低通道数
        self.inv5 = INV_block_affine(in_1=12, in_2=24) #[96,16,16]
        self.down5 = Down(in_channels * 4, in_channels * 8, pooling=True)

        self.up6 = Up(in_channels * 4, in_channels * 2, transpose=False)#[96,32,32]
        self.conv6 = nn.Conv2d(in_channels * 4, in_channels * 2, kernel_size=1)#[48,32,32]
        self.inv6 = INV_block_affine(in_1=6, in_2=12)#[48,32,32]
        self.down6 = Down(in_channels * 2, in_channels * 4, pooling=True)

        self.up7 = Up(in_channels * 2, in_channels, transpose=False)#[48,64,64]
        self.conv7 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)#[24,64,64]
        self.inv7 = INV_block_affine(in_1=3, in_2=6)#[24,64,64]
        self.down7 = Down(in_channels, in_channels * 2, pooling=True)

    def forward(self, x, rev=False):

        if not rev:
            inv1 = self.inv1(x)  # [24,64,64]
            down1 = self.down1(inv1)  # [48,32,32]

            inv2 = self.inv2(down1)  # [48,32,32]
            down2 = self.down2(inv2)  # [96,16,16]

            inv3 = self.inv3(down2)  # [96,16,16]
            down3 = self.down3(inv3)  # [192,8,8]

            inv4 = self.inv4(down3)  # [192,8,8]

            up5 = self.up5(inv4, inv3)  # [192,16,16]
            conv5 = self.conv5(up5)  # [96,16,16]
            inv5 = self.inv5(conv5)  # [96,16,16]
            

            up6 = self.up6(inv5, inv2)  # [96,32,32]
            conv6 = self.conv6(up6)  # [48,32,32]
            inv6 = self.inv6(conv6)  # [48,32,32]
           
            
            up7 = self.up7(inv6, inv1)  # [48,64,64]
            conv7 = self.conv7(up7)  # [24,64,64]
            inv7 = self.inv7(conv7)  # [24,64,64]
            out = inv7  # [24,64,64]

        else:
            inv7 = self.inv7(x, rev=True)  # [12,64,64]
            down7 = self.down7(inv7)  # [24,32,32]

            inv6 = self.inv6(down7, rev=True)  # [24,32,32]
            down6 = self.down6(inv6)  # [48,16,16]

            inv5 = self.inv5(down6, rev=True)  # [48,16,16]
            down5 = self.down5(inv5)  # [96,8,8]

            inv4 = self.inv4(down5, rev=True)  # [96,8,8]

            up3 = self.up3(inv4, inv5)  # [96,16,16]
           
            conv3 = self.conv3(up3)  # [48,16,16]
            inv3 = self.inv3(conv3, rev=True)  # [96,16,16]

            up2 = self.up2(inv3, inv6)  # [48,32,32]
            conv2 = self.conv2(up2)  # [24,32,32]
            inv2 = self.inv2(conv2)  # [48,32,32]
           

            up1 = self.up1(inv2, inv7)  # [24,64,64]
            conv1 = self.conv1(up1)  # [12,64,64]
            inv1 = self.inv1(conv1,rev=True)  # [24,64,64]
            out = inv1

        return out


#
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        super(Down, self).__init__()
        if pooling:
            self.down = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 1*1卷积扩充通道数
            )
        else:
            self.down = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = self.down(x)
        x = self.relu((x))
        return x


class Up(nn.Module):
    ''' up path
        conv_transpose => inv_block
    '''

    def __init__(self, in_ch, out_ch, transpose=False):
        super(Up, self).__init__()

        if transpose:
            self.Up = nn.ConvTranspose2d(in_ch, in_ch // 2, stride=2)
        else:
            # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.Up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
                                    nn.ReLU(inplace=True))

    def forward(self, x1, x2):
        '''
            conv output shape = (input_shape - Filter_shape + 2 * padding)/stride + 1
        '''

        x1 = self.Up(x1)

       # diffY = x2.size()[2] - x1.size()[2]
       # diffX = x2.size()[3] - x1.size()[3]
       # x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
       #                diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return x

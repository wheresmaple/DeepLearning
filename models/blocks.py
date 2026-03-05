import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#改进点1：RR-MADB替代密度连接模块DB
# 多尺度密集连接模块（MADB）+ ECA注意力
class MADB(nn.Module):
    def __init__(self, in_channels, growth_rate=64):
        super(MADB, self).__init__()
        self.in_channels = in_channels
        self.growth_rate = growth_rate

        self.conv3x3_1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.conv3x3_2 = nn.Conv2d(growth_rate, growth_rate, kernel_size=3, padding=1)

        self.eca = ECA(growth_rate * 2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        # 输出通道数映射回输入通道数
        self.out_conv = nn.Conv2d(growth_rate * 2, in_channels, kernel_size=1)

    def forward(self, x):
        # 第一级多尺度特征提取: 3x3 + 3x3串联
        f1_1 = self.relu(self.conv3x3_1(x))
        f1_2 = self.relu(self.conv3x3_2(f1_1))
        f1 = f1_1 + f1_2

        # 特征拼接
        # 第二级多尺度特征提取
        f2_1 = self.relu(self.conv3x3_1(f1))
        f2_2 = self.relu(self.conv3x3_2(f2_1))
        f2 = f2_1 + f2_2

        # ECA注意力
        eca_out = self.eca(torch.cat([f1, f2], dim=1))
        # 映射回输入通道数
        out = self.out_conv(eca_out)
        return out


# ECA通道注意力（轻量级）
class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECA, self).__init__()
        t = int(abs((np.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


# 多级残差模块（RR-MADB）
class RR_MADB(nn.Module):
    def __init__(self, in_channels, num_madb=4):
        super(RR_MADB, self).__init__()
        self.madb_blocks = nn.Sequential(*[MADB(in_channels) for _ in range(num_madb)])

    def forward(self, x):
        out = self.madb_blocks(x)
        return x + out  # 残差连接（输入输出通道数一致）


# 原始ESRGAN密集块（
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=64):
        super(DenseBlock, self).__init__()
        self.in_channels = in_channels
        self.growth_rate = growth_rate

        # 修正：卷积层输入通道数动态匹配
        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels + growth_rate, growth_rate, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        # 新增：压缩通道数回输入通道数
        self.transition = nn.Conv2d(in_channels + 2 * growth_rate, in_channels, kernel_size=1)

    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        concat1 = torch.cat([x, out1], dim=1)
        out2 = self.relu(self.conv2(concat1))
        concat2 = torch.cat([concat1, out2], dim=1)
        # 修正：压缩通道数，避免持续膨胀
        out = self.transition(concat2)
        return x + out  # 残差连接，保证通道数稳定
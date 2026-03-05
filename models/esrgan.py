import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import DenseBlock, RR_MADB

# ESRGAN主模型
class ESRGAN(nn.Module):
    def __init__(self, use_improved=True, in_channels=3, num_blocks=16, growth_rate=64):
        super(ESRGAN, self).__init__()
        self.use_improved = use_improved
        self.growth_rate = growth_rate
        self.base_channels = 64  # 基础通道数，保持全程一致

        # 初始卷积
        self.init_conv = nn.Conv2d(in_channels, self.base_channels, kernel_size=3, padding=1)

        # 核心块: 改进版(RR-MADB) / 原版(密集块)
        if use_improved:
            self.residual_blocks = RR_MADB(self.base_channels, num_madb=num_blocks)
        else:
            self.residual_blocks = nn.Sequential(*[
                DenseBlock(self.base_channels, growth_rate)
                for _ in range(num_blocks)
            ])

        # 上采样模块
        self.upsample = nn.Sequential(
            nn.Conv2d(self.base_channels, self.base_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),  #
            nn.LeakyReLU(0.2, inplace=True),
            # 输出3通道（RGB）
            nn.Conv2d(self.base_channels, 3, kernel_size=3, padding=1)
        )

        # 判别器
        self.discriminator = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, 3, H, W]
        x = self.init_conv(x)  # [B, 64, H, W]
        res = self.residual_blocks(x)  # [B, 64, H, W]（通道数不变）
        out = self.upsample(res + x)  # 全局残差
        return torch.clamp(out, 0, 1)

        #改进点2：相对判别器损失替代普通判别器损失
    def discriminate(self, x1, x2):
        # RaGAN判别器: 输出x1比x2更真实的概率
        concat = torch.cat([x1, x2], dim=1)  # [B, 6, H, W]
        if self.discriminator[0].in_channels != 6:
            self.discriminator[0] = nn.Conv2d(6, 64, kernel_size=3, padding=1).to(x1.device)
        return self.discriminator(concat)
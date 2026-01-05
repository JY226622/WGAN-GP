import torch
import torch.nn as nn

# ====================== Critic======================
class Critic(nn.Module):  # 评判器
    def __init__(self, img_size=96, channels=3):
        super(Critic, self).__init__()
        self.img_size = img_size  # 输入图片的尺寸（宽/高一致）
        self.channels = channels  # 输入图片的通道数（RGB为3，灰度图为1）

        def block(in_feat, out_feat, normalize=True):
            # 构建卷积块（下采样核心模块，提取图片特征）
            layers = [nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False)]  # 卷积：步长2，填充1，实现2倍下采样
            # WGAN-GP：Critic不使用BatchNorm（替换为InstanceNorm，避免破坏梯度惩罚的有效性）
            if normalize:
                layers.append(nn.InstanceNorm2d(out_feat, affine=True))  # 实例归一化：保留单张图片特征，适配梯度惩罚
            layers.append(nn.LeakyReLU(0.2, inplace=True))  # LeakyReLU：缓解梯度消失，保留负区间微弱梯度
            return layers

        # Critic主模型：通过多个卷积块逐步下采样，提取特征并输出连续分数
        self.model = nn.Sequential(
            *block(channels, 64, normalize=False),  # 3x96x96 → 64x48x48（第一次2倍下采样，不使用归一化）
            *block(64, 128),  # 64x48x48 → 128x24x24（第二次2倍下采样）
            *block(128, 256),  # 128x24x24 → 256x12x12（第三次2倍下采样）
            *block(256, 512),  # 256x12x12 → 512x6x6（第四次2倍下采样）
            nn.Conv2d(512, 1, 6, 1, 0, bias=False),  # 512x6x6 → 1x1x1（全局特征映射，输出单通道分数）
            # WGAN-GP：移除Sigmoid（Critic输出连续分数，非[0,1]概率值，无需激活函数映射）
        )

    def forward(self, img):
        validity = self.model(img)  # 输入图片，得到1x1x1的分数张量
        return validity.view(-1, 1).squeeze(1)  # 重塑为(batch_size,)的一维张量，便于计算损失
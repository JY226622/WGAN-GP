import torch
import torch.nn as nn

# ====================== Generator======================
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_size=96, channels=3):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim  # 潜在噪声向量的维度
        self.img_size = img_size      # 生成图片的尺寸（宽/高一致）
        self.channels = channels      # 生成图片的通道数（RGB为3，灰度图为1）
        self.init_size = img_size // 16  # 初始特征图尺寸（96//16=6，每次转置卷积使图片扩大2倍，共四次转置卷积，2×4=16.后续通过转置卷积上采样至目标尺寸）

        # 全连接层：将一维潜在噪声映射为高维特征向量，为后续转置卷积做准备
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 512 * self.init_size ** 2))  # 输出维度：batch_size × (512*6*6)

        def block(in_feat, out_feat, normalize=True):
            # 构建转置卷积块（上采样核心模块）
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 4, 2, 1, bias=False)]  # 转置卷积：步长2，填充1，实现2倍上采样
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))  # 批量归一化：稳定训练，缓解梯度消失
            layers.append(nn.ReLU(True))  # ReLU激活函数：引入非线性，保留正值特征
            return layers

        # 生成器主模型：通过多个转置卷积块逐步上采样，生成目标尺寸图片
        self.model = nn.Sequential(
            *block(512, 256),  # 512x6x6 → 256x12x12（第一次2倍上采样）
            *block(256, 128),  # 256x12x12 → 128x24x24（第二次2倍上采样）
            *block(128, 64),   # 128x24x24 → 64x48x48（第三次2倍上采样）
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),  # 64x48x48 → 3x96x96（第四次2倍上采样，得到目标尺寸）
            nn.Tanh()  # Tanh激活函数：将输出归一化到[-1, 1]，适配GAN训练的数据分布
        )

    def forward(self, z):
        z = z.view(z.size(0), -1)  # 将4维噪声张量(batch, latent_dim, 1, 1)展平为2维(batch, latent_dim)
        out = self.l1(z)  # 全连接层映射：2维 → 2维高维特征向量
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)  # 重塑为4维特征图(batch, 512, 6, 6)
        img = self.model(out)  # 传入转置卷积模型，生成目标尺寸图片
        return img  # 返回生成的图片张量(batch, channels, img_size, img_size)
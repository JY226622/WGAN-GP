"""
生成器网络 (Generator Network)
使用DCGAN架构，通过转置卷积逐步上采样生成图像
"""
import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    WGAN-GP 生成器
    
    Args:
        latent_dim: 潜在空间维度
        image_channels: 输出图像通道数
        feature_maps: 特征图基数
        image_size: 输出图像尺寸
    """
    def __init__(self, latent_dim=128, image_channels=3, feature_maps=64, image_size=64):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        # 计算初始特征图大小
        self.init_size = image_size // 16  # 对于64x64图像，初始为4x4
        
        # 线性层：将潜在向量映射到初始特征图
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, feature_maps * 8 * self.init_size * self.init_size),
            nn.BatchNorm1d(feature_maps * 8 * self.init_size * self.init_size),
            nn.ReLU(inplace=True)
        )
        
        # 转置卷积层：逐步上采样
        self.conv_blocks = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(inplace=True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(inplace=True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(feature_maps, image_channels, 4, 2, 1, bias=False),
            nn.Tanh()  # 输出范围 [-1, 1]
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, z):
        """
        前向传播
        
        Args:
            z: 潜在向量 (batch_size, latent_dim)
            
        Returns:
            生成的图像 (batch_size, image_channels, image_size, image_size)
        """
        # 线性变换
        out = self.fc(z)
        # 重塑为特征图
        out = out.view(out.size(0), -1, self.init_size, self.init_size)
        # 转置卷积上采样
        img = self.conv_blocks(out)
        return img


def test_generator():
    """测试生成器"""
    batch_size = 8
    latent_dim = 128
    
    # 创建生成器
    G = Generator(latent_dim=latent_dim, image_channels=3, feature_maps=64, image_size=64)
    
    # 生成随机噪声
    z = torch.randn(batch_size, latent_dim)
    
    # 生成图像
    fake_images = G(z)
    
    print(f"Generator output shape: {fake_images.shape}")
    print(f"Generator parameters: {sum(p.numel() for p in G.parameters()):,}")
    
    assert fake_images.shape == (batch_size, 3, 64, 64), "输出形状不正确"
    assert fake_images.min() >= -1 and fake_images.max() <= 1, "输出范围不在[-1, 1]"
    print("✓ Generator test passed!")


if __name__ == "__main__":
    test_generator()

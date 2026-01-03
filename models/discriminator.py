"""
判别器网络 (Discriminator/Critic Network)
在WGAN-GP中，判别器实际上是一个评分器(Critic)，输出实数而非概率
"""
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    WGAN-GP 判别器/评分器
    
    Args:
        image_channels: 输入图像通道数
        feature_maps: 特征图基数
        image_size: 输入图像尺寸
    """
    def __init__(self, image_channels=3, feature_maps=64, image_size=64):
        super(Discriminator, self).__init__()
        self.image_size = image_size
        
        # 卷积层：逐步下采样
        self.conv_blocks = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(image_channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.LayerNorm([feature_maps * 2, 16, 16]),  # 使用LayerNorm代替BatchNorm
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.LayerNorm([feature_maps * 4, 8, 8]),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 4x4
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.LayerNorm([feature_maps * 8, 4, 4]),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 全局平均池化和全连接层
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_maps * 8, 1)  # 输出单个实数评分
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
    
    def forward(self, img):
        """
        前向传播
        
        Args:
            img: 输入图像 (batch_size, image_channels, image_size, image_size)
            
        Returns:
            评分 (batch_size, 1) - 实数值，不使用sigmoid
        """
        features = self.conv_blocks(img)
        validity = self.fc(features)
        return validity


def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """
    计算梯度惩罚项
    
    Args:
        discriminator: 判别器模型
        real_samples: 真实样本
        fake_samples: 生成样本
        device: 计算设备
        
    Returns:
        gradient_penalty: 梯度惩罚值
    """
    batch_size = real_samples.size(0)
    
    # 随机权重用于插值
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    
    # 计算插值样本
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    
    # 计算判别器对插值样本的输出
    d_interpolates = discriminator(interpolates)
    
    # 计算梯度
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # 重塑梯度
    gradients = gradients.view(batch_size, -1)
    
    # 计算梯度的L2范数
    gradient_norm = gradients.norm(2, dim=1)
    
    # 计算梯度惩罚
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    
    return gradient_penalty


def test_discriminator():
    """测试判别器"""
    batch_size = 8
    image_size = 64
    
    # 创建判别器
    D = Discriminator(image_channels=3, feature_maps=64, image_size=image_size)
    
    # 创建随机图像
    images = torch.randn(batch_size, 3, image_size, image_size)
    
    # 获取评分
    scores = D(images)
    
    print(f"Discriminator output shape: {scores.shape}")
    print(f"Discriminator parameters: {sum(p.numel() for p in D.parameters()):,}")
    
    assert scores.shape == (batch_size, 1), "输出形状不正确"
    print("✓ Discriminator test passed!")
    
    # 测试梯度惩罚
    real_samples = torch.randn(batch_size, 3, image_size, image_size)
    fake_samples = torch.randn(batch_size, 3, image_size, image_size)
    gp = compute_gradient_penalty(D, real_samples, fake_samples, 'cpu')
    
    print(f"Gradient penalty: {gp.item():.4f}")
    print("✓ Gradient penalty test passed!")


if __name__ == "__main__":
    test_discriminator()

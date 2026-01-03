"""
WGAN-GP 模型封装
整合生成器和判别器，提供训练和生成接口
"""
import torch
import torch.nn as nn
from .generator import Generator
from .discriminator import Discriminator, compute_gradient_penalty


class WGANGP:
    """
    WGAN-GP 模型封装类
    
    Args:
        config: 配置字典
        device: 计算设备
    """
    def __init__(self, config, device):
        self.config = config
        self.device = device
        
        # 创建生成器和判别器
        self.generator = Generator(
            latent_dim=config['model']['latent_dim'],
            image_channels=config['data']['image_channels'],
            feature_maps=config['model']['feature_maps_g'],
            image_size=config['data']['image_size']
        ).to(device)
        
        self.discriminator = Discriminator(
            image_channels=config['data']['image_channels'],
            feature_maps=config['model']['feature_maps_d'],
            image_size=config['data']['image_size']
        ).to(device)
        
        # 优化器
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=config['training']['optimizer']['lr_g'],
            betas=(config['training']['optimizer']['beta1'], 
                   config['training']['optimizer']['beta2'])
        )
        
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config['training']['optimizer']['lr_d'],
            betas=(config['training']['optimizer']['beta1'], 
                   config['training']['optimizer']['beta2'])
        )
        
        # 超参数
        self.lambda_gp = config['training']['wgan_gp']['lambda_gp']
        self.n_critic = config['training']['n_critic']
        self.latent_dim = config['model']['latent_dim']
    
    def train_discriminator(self, real_images):
        """
        训练判别器
        
        Args:
            real_images: 真实图像批次
            
        Returns:
            d_loss: 判别器损失
            wasserstein_distance: Wasserstein距离
        """
        batch_size = real_images.size(0)
        
        self.optimizer_D.zero_grad()
        
        # 生成假图像
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_images = self.generator(z).detach()
        
        # 判别器对真实和假图像的评分
        real_validity = self.discriminator(real_images)
        fake_validity = self.discriminator(fake_images)
        
        # 计算梯度惩罚
        gradient_penalty = compute_gradient_penalty(
            self.discriminator, real_images, fake_images, self.device
        )
        
        # 计算判别器损失
        # WGAN loss: E[D(fake)] - E[D(real)] + λ * gradient_penalty
        d_loss = (
            torch.mean(fake_validity) - torch.mean(real_validity) + 
            self.lambda_gp * gradient_penalty
        )
        
        d_loss.backward()
        self.optimizer_D.step()
        
        # Wasserstein距离估计
        wasserstein_distance = torch.mean(real_validity) - torch.mean(fake_validity)
        
        return d_loss.item(), wasserstein_distance.item()
    
    def train_generator(self):
        """
        训练生成器
        
        Returns:
            g_loss: 生成器损失
        """
        self.optimizer_G.zero_grad()
        
        # 从配置中获取批次大小
        batch_size = self.config['training']['batch_size']
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_images = self.generator(z)
        
        # 计算生成器损失
        # WGAN generator loss: -E[D(fake)]
        fake_validity = self.discriminator(fake_images)
        g_loss = -torch.mean(fake_validity)
        
        g_loss.backward()
        self.optimizer_G.step()
        
        return g_loss.item()
    
    def generate(self, num_images=64, batch_size=64):
        """
        生成图像
        
        Args:
            num_images: 生成图像数量
            batch_size: 批次大小
            
        Returns:
            生成的图像列表
        """
        self.generator.eval()
        generated_images = []
        
        with torch.no_grad():
            num_batches = (num_images + batch_size - 1) // batch_size
            
            for i in range(num_batches):
                current_batch_size = min(batch_size, num_images - i * batch_size)
                z = torch.randn(current_batch_size, self.latent_dim, device=self.device)
                fake_images = self.generator(z)
                generated_images.append(fake_images.cpu())
        
        self.generator.train()
        return torch.cat(generated_images, dim=0)[:num_images]
    
    def save_checkpoint(self, path, epoch, d_loss, g_loss):
        """
        保存模型检查点
        
        Args:
            path: 保存路径
            epoch: 当前轮数
            d_loss: 判别器损失
            g_loss: 生成器损失
        """
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'd_loss': d_loss,
            'g_loss': g_loss,
            'config': self.config
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """
        加载模型检查点
        
        Args:
            path: 检查点路径
            
        Returns:
            epoch: 训练轮数
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        
        return checkpoint['epoch']


def test_wgan_gp():
    """测试WGAN-GP模型"""
    # 简单配置
    config = {
        'model': {
            'latent_dim': 128,
            'feature_maps_g': 64,
            'feature_maps_d': 64
        },
        'data': {
            'image_channels': 3,
            'image_size': 64
        },
        'training': {
            'optimizer': {
                'lr_g': 0.0001,
                'lr_d': 0.0001,
                'beta1': 0.0,
                'beta2': 0.9
            },
            'wgan_gp': {
                'lambda_gp': 10
            },
            'n_critic': 5
        }
    }
    
    device = torch.device('cpu')
    model = WGANGP(config, device)
    
    # 测试判别器训练
    real_images = torch.randn(8, 3, 64, 64).to(device)
    d_loss, wd = model.train_discriminator(real_images)
    print(f"D loss: {d_loss:.4f}, Wasserstein distance: {wd:.4f}")
    
    # 测试生成器训练
    g_loss = model.train_generator()
    print(f"G loss: {g_loss:.4f}")
    
    # 测试生成
    generated = model.generate(num_images=16, batch_size=8)
    print(f"Generated images shape: {generated.shape}")
    
    print("✓ WGAN-GP model test passed!")


if __name__ == "__main__":
    test_wgan_gp()

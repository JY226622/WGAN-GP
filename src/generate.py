"""
图像生成脚本
使用训练好的WGAN-GP模型生成新的二次元图像
"""
import os
import sys
import argparse
from tqdm import tqdm

import torch

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.wgan_gp import WGANGP
from src.utils import (
    load_config,
    set_seed,
    get_device,
    save_single_image,
    save_image_grid
)


def generate_images(model, num_images, batch_size, output_dir, save_format='png'):
    """
    批量生成图像
    
    Args:
        model: WGAN-GP模型
        num_images: 生成图像数量
        batch_size: 批次大小
        output_dir: 输出目录
        save_format: 保存格式
    """
    print(f"生成 {num_images} 张图像...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 批量生成
    num_batches = (num_images + batch_size - 1) // batch_size
    image_count = 0
    
    for batch_idx in tqdm(range(num_batches), desc="生成中"):
        current_batch_size = min(batch_size, num_images - batch_idx * batch_size)
        
        # 生成图像
        z = torch.randn(current_batch_size, model.latent_dim, device=model.device)
        with torch.no_grad():
            fake_images = model.generator(z)
        
        # 保存每张图像
        for i in range(current_batch_size):
            save_path = os.path.join(
                output_dir, 
                f'generated_{image_count:06d}.{save_format}'
            )
            save_single_image(fake_images[i], save_path, normalize=True)
            image_count += 1
    
    print(f"✓ 已生成 {image_count} 张图像，保存在: {output_dir}")
    
    # 生成预览网格
    preview_images = model.generate(num_images=64, batch_size=64)
    preview_path = os.path.join(output_dir, 'preview_grid.png')
    save_image_grid(preview_images, preview_path, nrow=8)
    print(f"✓ 预览网格已保存: {preview_path}")


def interpolate_latent(model, num_steps=10, output_dir='interpolation'):
    """
    在潜在空间进行插值，生成过渡图像
    
    Args:
        model: WGAN-GP模型
        num_steps: 插值步数
        output_dir: 输出目录
    """
    print(f"\n潜在空间插值 ({num_steps} 步)...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成两个随机潜在向量
    z1 = torch.randn(1, model.latent_dim, device=model.device)
    z2 = torch.randn(1, model.latent_dim, device=model.device)
    
    interpolated_images = []
    
    with torch.no_grad():
        for i in range(num_steps):
            # 线性插值
            alpha = i / (num_steps - 1)
            z = (1 - alpha) * z1 + alpha * z2
            
            # 生成图像
            fake_image = model.generator(z)
            interpolated_images.append(fake_image)
            
            # 保存单张图像
            save_path = os.path.join(output_dir, f'interp_{i:03d}.png')
            save_single_image(fake_image[0], save_path, normalize=True)
    
    # 保存插值序列网格
    all_images = torch.cat(interpolated_images, dim=0)
    grid_path = os.path.join(output_dir, 'interpolation_grid.png')
    save_image_grid(all_images, grid_path, nrow=num_steps)
    
    print(f"✓ 插值序列已保存: {output_dir}")


def random_sampling(model, num_samples=100, output_dir='random_samples'):
    """
    随机采样生成多样化的图像
    
    Args:
        model: WGAN-GP模型
        num_samples: 采样数量
        output_dir: 输出目录
    """
    print(f"\n随机采样 ({num_samples} 张)...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    fake_images = model.generate(num_images=num_samples, batch_size=64)
    
    # 保存每张图像
    for i in range(num_samples):
        save_path = os.path.join(output_dir, f'sample_{i:04d}.png')
        save_single_image(fake_images[i], save_path, normalize=True)
    
    # 保存网格
    grid_path = os.path.join(output_dir, 'samples_grid.png')
    save_image_grid(fake_images[:64], grid_path, nrow=8)
    
    print(f"✓ 随机样本已保存: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate anime images using trained WGAN-GP')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--num_images', type=int, default=1000,
                        help='Number of images to generate')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for generation')
    parser.add_argument('--output', type=str, default='data/generated',
                        help='Output directory')
    parser.add_argument('--mode', type=str, default='generate',
                        choices=['generate', 'interpolate', 'sample', 'all'],
                        help='Generation mode')
    parser.add_argument('--interpolate_steps', type=int, default=10,
                        help='Number of interpolation steps')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of random samples')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("WGAN-GP 图像生成 - 二次元图像数据增强")
    print("=" * 80)
    
    # 加载配置
    config = load_config(args.config)
    set_seed(config['device']['seed'])
    
    # 获取设备
    device = get_device(config)
    
    # 创建模型
    print("\n加载模型...")
    model = WGANGP(config, device)
    
    # 加载训练好的模型
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在 - {args.model}")
        return
    
    print(f"从检查点加载: {args.model}")
    epoch = model.load_checkpoint(args.model)
    print(f"模型训练至 epoch {epoch}")
    
    # 设置为评估模式
    model.generator.eval()
    
    # 根据模式执行生成
    if args.mode == 'generate' or args.mode == 'all':
        # 获取保存格式，如果config中没有则使用默认值
        save_format = config.get('generation', {}).get('save_format', 'png')
        generate_images(
            model, 
            args.num_images, 
            args.batch_size, 
            args.output,
            save_format=save_format
        )
    
    if args.mode == 'interpolate' or args.mode == 'all':
        interp_dir = os.path.join(args.output, 'interpolation')
        interpolate_latent(model, args.interpolate_steps, interp_dir)
    
    if args.mode == 'sample' or args.mode == 'all':
        sample_dir = os.path.join(args.output, 'random_samples')
        random_sampling(model, args.num_samples, sample_dir)
    
    print("\n" + "=" * 80)
    print("图像生成完成!")
    print(f"输出目录: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()

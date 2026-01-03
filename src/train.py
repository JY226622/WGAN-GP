"""
WGAN-GP 训练脚本
"""
import os
import sys
import argparse
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.wgan_gp import WGANGP
from src.utils import (
    load_config, 
    set_seed, 
    get_device, 
    get_dataloader,
    save_image_grid,
    AverageMeter,
    create_directories,
    count_parameters
)


def train_one_epoch(model, dataloader, epoch, writer, config):
    """
    训练一个epoch
    
    Args:
        model: WGAN-GP模型
        dataloader: 数据加载器
        epoch: 当前epoch
        writer: TensorBoard写入器
        config: 配置字典
    """
    # 损失记录器
    d_loss_meter = AverageMeter()
    g_loss_meter = AverageMeter()
    wd_meter = AverageMeter()
    
    # 进度条
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, real_images in enumerate(pbar):
        real_images = real_images.to(model.device)
        
        # 训练判别器
        d_loss, wd = model.train_discriminator(real_images)
        d_loss_meter.update(d_loss)
        wd_meter.update(wd)
        
        # 每n_critic步训练一次生成器
        if batch_idx % config['training']['n_critic'] == 0:
            g_loss = model.train_generator()
            g_loss_meter.update(g_loss)
        
        # 更新进度条
        pbar.set_postfix({
            'D_loss': f'{d_loss_meter.avg:.4f}',
            'G_loss': f'{g_loss_meter.avg:.4f}',
            'WD': f'{wd_meter.avg:.4f}'
        })
        
        # 记录到TensorBoard
        global_step = epoch * len(dataloader) + batch_idx
        if batch_idx % config['logging']['log_interval'] == 0:
            writer.add_scalar('Loss/Discriminator', d_loss, global_step)
            writer.add_scalar('Loss/Generator', g_loss_meter.val, global_step)
            writer.add_scalar('Metric/Wasserstein_Distance', wd, global_step)
    
    return d_loss_meter.avg, g_loss_meter.avg, wd_meter.avg


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train WGAN-GP on anime images')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    print("=" * 80)
    print("WGAN-GP 训练 - 二次元图像生成与数据增强")
    print("=" * 80)
    
    # 设置随机种子
    set_seed(config['device']['seed'])
    
    # 创建必要的目录
    create_directories(config)
    
    # 获取设备
    device = get_device(config)
    
    # 创建模型
    print("\n创建模型...")
    model = WGANGP(config, device)
    
    # 统计参数
    g_params, g_trainable = count_parameters(model.generator)
    d_params, d_trainable = count_parameters(model.discriminator)
    print(f"生成器参数: {g_params:,} (可训练: {g_trainable:,})")
    print(f"判别器参数: {d_params:,} (可训练: {d_trainable:,})")
    print(f"总参数: {g_params + d_params:,}")
    
    # 加载检查点（如果需要）
    start_epoch = 0
    if args.resume:
        print(f"\n从检查点恢复: {args.resume}")
        start_epoch = model.load_checkpoint(args.resume)
        print(f"从 epoch {start_epoch} 继续训练")
    
    # 创建数据加载器
    print("\n加载数据集...")
    try:
        dataloader = get_dataloader(config, shuffle=True)
        print(f"数据集大小: {len(dataloader.dataset)}")
        print(f"批次数量: {len(dataloader)}")
    except Exception as e:
        print(f"错误: 无法加载数据集 - {e}")
        print("提示: 请确保数据已放置在 data/processed/ 目录下")
        return
    
    # 创建TensorBoard写入器
    writer = SummaryWriter(config['logging']['tensorboard_dir'])
    
    # 训练循环
    print("\n开始训练...")
    print("=" * 80)
    
    num_epochs = config['training']['num_epochs']
    save_interval = config['logging']['save_interval']
    checkpoint_dir = config['logging']['checkpoint_dir']
    
    best_wd = float('inf')
    
    for epoch in range(start_epoch, num_epochs):
        # 训练一个epoch
        d_loss, g_loss, wd = train_one_epoch(model, dataloader, epoch, writer, config)
        
        # 打印epoch统计
        print(f"\nEpoch {epoch}/{num_epochs - 1}")
        print(f"  D Loss: {d_loss:.4f}")
        print(f"  G Loss: {g_loss:.4f}")
        print(f"  Wasserstein Distance: {wd:.4f}")
        
        # 生成样本图像
        if config['logging']['generate_samples']:
            num_samples = config['logging']['num_sample_images']
            fake_images = model.generate(num_images=num_samples, batch_size=num_samples)
            
            # 保存到TensorBoard
            writer.add_images('Generated_Images', 
                              (fake_images + 1) / 2,  # 归一化到[0, 1]
                              epoch)
            
            # 保存到文件
            sample_dir = os.path.join(config['data']['generated_dir'], 'samples')
            os.makedirs(sample_dir, exist_ok=True)
            save_image_grid(
                fake_images[:64], 
                os.path.join(sample_dir, f'epoch_{epoch:04d}.png'),
                nrow=8
            )
        
        # 保存检查点
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, 
                f'checkpoint_epoch_{epoch:04d}.pth'
            )
            model.save_checkpoint(checkpoint_path, epoch, d_loss, g_loss)
            print(f"  ✓ 检查点已保存: {checkpoint_path}")
        
        # 保存最佳模型（基于Wasserstein距离）
        if abs(wd) < best_wd:
            best_wd = abs(wd)
            best_path = os.path.join(checkpoint_dir, 'best_model.pth')
            model.save_checkpoint(best_path, epoch, d_loss, g_loss)
            print(f"  ✓ 最佳模型已更新 (WD: {wd:.4f})")
    
    # 保存最终模型
    final_path = os.path.join(checkpoint_dir, 'final_model.pth')
    model.save_checkpoint(final_path, num_epochs - 1, d_loss, g_loss)
    
    print("\n" + "=" * 80)
    print("训练完成!")
    print(f"最佳模型保存在: {best_path}")
    print(f"最终模型保存在: {final_path}")
    print(f"TensorBoard日志: {config['logging']['tensorboard_dir']}")
    print("=" * 80)
    
    writer.close()


if __name__ == "__main__":
    main()

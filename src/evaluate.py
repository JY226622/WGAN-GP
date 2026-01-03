"""
模型评估脚本
计算FID、IS等评估指标
"""
import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import inception_v3
from scipy import linalg
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.wgan_gp import WGANGP
from src.utils import load_config, set_seed, get_device, AnimeDataset


def get_inception_features(images, model, device, batch_size=32):
    """
    使用Inception网络提取特征
    
    Args:
        images: 图像数据集或目录
        model: Inception模型
        device: 计算设备
        batch_size: 批次大小
        
    Returns:
        features: 提取的特征
    """
    model.eval()
    features_list = []
    
    # 如果是目录，创建数据集
    if isinstance(images, str):
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        dataset = AnimeDataset(images, image_size=299, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    else:
        # 假设是张量
        dataloader = DataLoader(images, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="提取特征"):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device)
            
            # 调整大小到299x299（Inception输入大小）
            if batch.shape[2] != 299:
                batch = torch.nn.functional.interpolate(
                    batch, size=(299, 299), mode='bilinear', align_corners=False
                )
            
            # 提取特征
            features = model(batch)
            features_list.append(features.cpu().numpy())
    
    return np.concatenate(features_list, axis=0)


def calculate_fid(real_features, fake_features):
    """
    计算Fréchet Inception Distance
    
    Args:
        real_features: 真实图像特征
        fake_features: 生成图像特征
        
    Returns:
        fid: FID分数
    """
    # 计算均值和协方差
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    
    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)
    
    # 计算差值
    diff = mu_real - mu_fake
    
    # 计算协方差矩阵的平方根
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
    
    # 处理数值误差
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # 计算FID
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    
    return fid


def calculate_inception_score(features, splits=10):
    """
    计算Inception Score
    
    注意：这是一个简化的实现。完整的IS计算需要使用Inception-v3的
    分类输出（1000个类别的softmax概率）。
    
    Args:
        features: Inception-v3的池化层特征（不是分类概率）
        splits: 分割数量
        
    Returns:
        is_mean: IS均值（注意：此实现仅作演示，结果可能不准确）
        is_std: IS标准差
    """
    # 警告：这是简化实现
    print("警告: 这是简化的IS计算实现，结果仅供参考。")
    print("完整的IS计算需要使用Inception-v3的分类输出。")
    
    # 归一化特征作为伪概率（仅用于演示）
    # 实际应该使用softmax分类概率
    probs = np.abs(features) / (np.abs(features).sum(axis=1, keepdims=True) + 1e-10)
    
    # 分割计算
    split_scores = []
    N = probs.shape[0]
    
    for k in range(splits):
        part = probs[k * N // splits: (k + 1) * N // splits]
        
        # KL散度
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i]
            scores.append(np.sum(pyx * (np.log(pyx + 1e-10) - np.log(py + 1e-10))))
        
        split_scores.append(np.exp(np.mean(scores)))
    
    return np.mean(split_scores), np.std(split_scores)


def evaluate_model(config, model_path, real_dir, num_samples=1000):
    """
    评估WGAN-GP模型
    
    Args:
        config: 配置字典
        model_path: 模型路径
        real_dir: 真实图像目录
        num_samples: 生成样本数量
    """
    device = get_device(config)
    
    # 加载模型
    print("加载模型...")
    model = WGANGP(config, device)
    epoch = model.load_checkpoint(model_path)
    print(f"模型训练至 epoch {epoch}")
    
    # 创建Inception模型
    print("加载Inception模型...")
    inception = inception_v3(pretrained=True, transform_input=False)
    inception = inception.to(device)
    inception.fc = torch.nn.Identity()  # 移除最后的分类层
    
    # 生成图像
    print(f"生成 {num_samples} 张图像...")
    fake_images = model.generate(num_images=num_samples, batch_size=64)
    
    # 提取真实图像特征
    print("提取真实图像特征...")
    real_features = get_inception_features(real_dir, inception, device)
    
    # 提取生成图像特征
    print("提取生成图像特征...")
    fake_features = get_inception_features(fake_images, inception, device)
    
    # 计算FID
    print("\n计算FID...")
    fid_score = calculate_fid(real_features, fake_features)
    print(f"FID Score: {fid_score:.2f}")
    
    # 计算IS
    print("\n计算IS...")
    is_mean, is_std = calculate_inception_score(fake_features)
    print(f"IS Score: {is_mean:.2f} ± {is_std:.2f}")
    
    return {
        'fid': fid_score,
        'is_mean': is_mean,
        'is_std': is_std
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate WGAN-GP model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--real_dir', type=str, required=True,
                        help='Directory containing real images')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to generate for evaluation')
    parser.add_argument('--metric', type=str, default='all',
                        choices=['fid', 'is', 'all'],
                        help='Evaluation metric')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("WGAN-GP 模型评估")
    print("=" * 80)
    
    # 加载配置
    config = load_config(args.config)
    set_seed(config['device']['seed'])
    
    # 评估模型
    results = evaluate_model(
        config,
        args.model,
        args.real_dir,
        args.num_samples
    )
    
    print("\n" + "=" * 80)
    print("评估结果:")
    print(f"  FID Score: {results['fid']:.2f}")
    print(f"  IS Score: {results['is_mean']:.2f} ± {results['is_std']:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

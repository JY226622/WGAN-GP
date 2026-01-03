"""
工具函数
包含数据加载、图像保存、可视化等辅助功能
"""
import os
import yaml
import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


def set_seed(seed=42):
    """
    设置随机种子以保证可重复性
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保卷积算法的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, save_path):
    """
    保存配置文件
    
    Args:
        config: 配置字典
        save_path: 保存路径
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)


class AnimeDataset(Dataset):
    """
    二次元图像数据集
    
    Args:
        root_dir: 数据根目录
        image_size: 图像尺寸
        transform: 数据变换
    """
    def __init__(self, root_dir, image_size=64, transform=None):
        self.root_dir = root_dir
        self.image_size = image_size
        
        # 获取所有图像文件
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        self.image_files = [
            f for f in os.listdir(root_dir)
            if os.path.splitext(f)[1].lower() in valid_extensions
        ]
        
        # 默认变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化到[-1, 1]
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image


def get_dataloader(config, shuffle=True):
    """
    创建数据加载器
    
    Args:
        config: 配置字典
        shuffle: 是否打乱数据
        
    Returns:
        DataLoader对象
    """
    dataset = AnimeDataset(
        root_dir=config['data']['processed_dir'],
        image_size=config['data']['image_size']
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=shuffle,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    return dataloader


def save_image_grid(images, save_path, nrow=8, normalize=True):
    """
    保存图像网格
    
    Args:
        images: 图像张量 (N, C, H, W)
        save_path: 保存路径
        nrow: 每行图像数量
        normalize: 是否从[-1,1]归一化到[0,1]
    """
    if normalize:
        images = (images + 1) / 2  # 从[-1, 1]转换到[0, 1]
    
    # 裁剪到有效范围
    images = torch.clamp(images, 0, 1)
    
    # 计算网格布局
    batch_size = images.size(0)
    ncol = (batch_size + nrow - 1) // nrow
    
    # 创建图像网格
    fig, axes = plt.subplots(ncol, nrow, figsize=(nrow * 2, ncol * 2))
    axes = axes.flatten() if batch_size > 1 else [axes]
    
    for idx in range(batch_size):
        img = images[idx].permute(1, 2, 0).cpu().numpy()
        axes[idx].imshow(img)
        axes[idx].axis('off')
    
    # 隐藏多余的子图
    for idx in range(batch_size, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_single_image(image, save_path, normalize=True):
    """
    保存单张图像
    
    Args:
        image: 图像张量 (C, H, W)
        save_path: 保存路径
        normalize: 是否从[-1,1]归一化到[0,1]
    """
    if normalize:
        image = (image + 1) / 2
    
    image = torch.clamp(image, 0, 1)
    image = image.permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    
    Image.fromarray(image).save(save_path)


def denormalize(tensor):
    """
    反归一化：从[-1, 1]转换到[0, 1]
    
    Args:
        tensor: 归一化的张量
        
    Returns:
        反归一化的张量
    """
    return (tensor + 1) / 2


def get_device(config):
    """
    获取计算设备
    
    Args:
        config: 配置字典
        
    Returns:
        torch.device对象
    """
    if config['device']['use_cuda'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['device']['gpu_ids'][0]}")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def count_parameters(model):
    """
    统计模型参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        total_params: 总参数数
        trainable_params: 可训练参数数
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


class AverageMeter:
    """
    计算和存储平均值和当前值
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_directories(config):
    """
    创建必要的目录
    
    Args:
        config: 配置字典
    """
    dirs = [
        config['data']['raw_dir'],
        config['data']['processed_dir'],
        config['data']['generated_dir'],
        config['logging']['checkpoint_dir'],
        config['logging']['tensorboard_dir'],
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)


if __name__ == "__main__":
    # 测试工具函数
    print("Testing utility functions...")
    
    # 测试种子设置
    set_seed(42)
    print("✓ Seed set successfully")
    
    # 测试配置加载
    config_path = "config/config.yaml"
    if os.path.exists(config_path):
        config = load_config(config_path)
        print(f"✓ Config loaded: {len(config)} sections")
    else:
        print("⚠ Config file not found")
    
    print("✓ All utility tests passed!")

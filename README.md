# WGAN-GP 二次元图片生成与数据增强

> 2025/2026秋季人工智能原理与应用课程大作业

## 项目简介

本项目基于 **WGAN-GP (Wasserstein GAN with Gradient Penalty)** 算法实现二次元（动漫）图片的生成与数据增强。WGAN-GP 是一种改进的生成对抗网络（GAN）算法，通过引入梯度惩罚项来稳定训练过程，能够生成高质量的图像，特别适合用于数据增强任务。

### 项目目标

1. **理解 WGAN-GP 原理**：深入学习 Wasserstein 距离和梯度惩罚机制
2. **实现数据增强**：使用 WGAN-GP 生成新的二次元图片，扩充训练数据集
3. **评估增强效果**：验证生成图片的质量和多样性
4. **应用场景展示**：展示 WGAN-GP 在实际数据增强任务中的应用价值

## 目录结构

```
WGAN-GP/
├── README.md           # 项目说明文档
├── LICENSE             # 开源许可证
├── requirements.txt    # Python依赖包
├── .gitignore         # Git忽略文件配置
├── config/            # 配置文件目录
│   └── config.yaml    # 训练配置参数
├── data/              # 数据集目录
│   ├── raw/           # 原始数据
│   ├── processed/     # 预处理后的数据
│   └── generated/     # 生成的图片
├── docs/              # 文档目录
│   ├── 技术文档.md    # 技术实现细节
│   └── 项目报告.md    # 项目总结报告
├── models/            # 模型定义
│   ├── generator.py   # 生成器网络
│   ├── discriminator.py # 判别器网络
│   └── wgan_gp.py     # WGAN-GP模型
├── src/               # 源代码目录
│   ├── train.py       # 训练脚本
│   ├── generate.py    # 图片生成脚本
│   ├── evaluate.py    # 评估脚本
│   └── utils.py       # 工具函数
├── notebooks/         # Jupyter笔记本
│   └── demo.ipynb     # 演示笔记本
└── checkpoints/       # 模型检查点
    └── saved_models/  # 保存的模型
```

## WGAN-GP 原理简介

### 什么是 GAN？

生成对抗网络（Generative Adversarial Network, GAN）由两个神经网络组成：
- **生成器（Generator）**：从随机噪声生成假图片
- **判别器（Discriminator）**：区分真实图片和生成图片

两者通过对抗训练不断提升，最终生成器能够生成以假乱真的图片。

### WGAN-GP 的改进

传统 GAN 训练不稳定，容易出现模式崩溃。WGAN-GP 通过以下改进解决这些问题：

1. **Wasserstein 距离**：使用 Wasserstein 距离代替 JS 散度，提供更稳定的梯度
2. **梯度惩罚**：通过梯度惩罚项约束判别器，替代权重裁剪
3. **训练稳定性**：更容易收敛，生成质量更高

数学表达式：

```
L_D = E[D(x_fake)] - E[D(x_real)] + λ·E[(||∇D(x_hat)||₂ - 1)²]
L_G = -E[D(x_fake)]
```

其中：
- `L_D`: 判别器损失
- `L_G`: 生成器损失
- `λ`: 梯度惩罚系数（通常为10）
- `x_hat`: 真实和生成样本的插值

## 环境配置

### 系统要求

- Python 3.8+
- CUDA 11.0+ (使用GPU加速)
- 至少 8GB RAM
- 推荐 GPU: NVIDIA GTX 1060 或更高

### 安装依赖

```bash
# 克隆项目
git clone https://github.com/JY226622/WGAN-GP.git
cd WGAN-GP

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖包
pip install -r requirements.txt
```

### 主要依赖

- **PyTorch**: 深度学习框架
- **torchvision**: 图像处理和预训练模型
- **numpy**: 数值计算
- **Pillow**: 图像处理
- **matplotlib**: 可视化
- **tensorboard**: 训练监控

## 使用方法

### 1. 准备数据集

将二次元图片数据集放置在 `data/raw/` 目录下：

```bash
# 下载数据集（例如：Anime Face Dataset）
# 或使用自己收集的二次元图片

# 数据预处理
python src/preprocess.py --input data/raw/ --output data/processed/
```

### 2. 训练模型

```bash
# 使用默认配置训练
python src/train.py

# 使用自定义配置
python src/train.py --config config/config.yaml

# 从检查点恢复训练
python src/train.py --resume checkpoints/saved_models/model_epoch_100.pth
```

训练参数（在 `config/config.yaml` 中配置）：

```yaml
# 基本设置
batch_size: 64
image_size: 64
latent_dim: 128
num_epochs: 200

# 优化器设置
lr_g: 0.0001  # 生成器学习率
lr_d: 0.0001  # 判别器学习率
beta1: 0.0
beta2: 0.9

# WGAN-GP 设置
lambda_gp: 10  # 梯度惩罚系数
n_critic: 5    # 判别器训练次数

# 其他设置
save_interval: 10  # 保存间隔
log_interval: 100  # 日志间隔
```

### 3. 生成图片

```bash
# 生成新图片
python src/generate.py --model checkpoints/saved_models/best_model.pth \
                       --num_images 1000 \
                       --output data/generated/

# 批量生成用于数据增强
python src/generate.py --model checkpoints/saved_models/best_model.pth \
                       --num_images 5000 \
                       --batch_size 128
```

### 4. 评估模型

```bash
# 评估生成质量
python src/evaluate.py --model checkpoints/saved_models/best_model.pth

# 计算 FID 分数（Fréchet Inception Distance）
python src/evaluate.py --metric fid --real_dir data/processed/ --fake_dir data/generated/

# 计算 IS 分数（Inception Score）
python src/evaluate.py --metric is --fake_dir data/generated/
```

### 5. 可视化训练过程

```bash
# 启动 TensorBoard
tensorboard --logdir=runs/

# 在浏览器中打开 http://localhost:6006
```

## 数据增强应用

### 为什么需要数据增强？

在深度学习任务中，数据量往往是影响模型性能的关键因素。然而：
- 收集高质量标注数据成本高昂
- 某些领域数据稀缺（如医学图像）
- 数据不平衡问题

### WGAN-GP 在数据增强中的优势

1. **高质量生成**：生成的图片质量高，细节丰富
2. **多样性好**：通过采样不同的噪声向量，生成多样化的图片
3. **可控性强**：可以引导生成特定风格的图片
4. **避免过拟合**：扩充数据集，提升模型泛化能力

### 实际应用流程

```python
# 示例代码：使用生成的图片进行数据增强
import torch
from models.wgan_gp import Generator

# 加载训练好的生成器
generator = Generator(latent_dim=128)
generator.load_state_dict(torch.load('checkpoints/saved_models/best_model.pth'))
generator.eval()

# 生成新图片
num_augment = 1000
with torch.no_grad():
    for i in range(num_augment):
        z = torch.randn(1, 128)  # 随机噪声
        fake_img = generator(z)
        save_image(fake_img, f'data/augmented/img_{i}.png')

# 将生成的图片加入训练集
# 继续训练下游任务模型（分类、检测等）
```

## 实验结果

### 训练曲线

训练过程中，观察以下指标：
- **生成器损失（G Loss）**：应逐渐下降并稳定
- **判别器损失（D Loss）**：保持在一个合理范围内
- **Wasserstein 距离**：衡量真实分布和生成分布的距离

### 生成样本质量

评估指标：
- **FID (Fréchet Inception Distance)**：越低越好，衡量生成图片与真实图片的分布距离
- **IS (Inception Score)**：越高越好，衡量生成图片的质量和多样性
- **人工评估**：主观评价生成图片的真实感

### 数据增强效果

对比实验：
1. **基线模型**：仅使用原始数据集训练
2. **增强模型**：使用原始+生成数据训练

预期提升：
- 分类准确率提升 2-5%
- 更好的泛化性能
- 减少过拟合现象

## 常见问题

### Q1: 训练不稳定怎么办？

- 调整学习率（降低至 0.0001 或更低）
- 增加判别器训练次数（n_critic）
- 调整梯度惩罚系数（lambda_gp）
- 使用 Spectral Normalization

### Q2: 生成图片模糊？

- 增加训练轮数
- 使用更大的网络（增加层数或通道数）
- 调整批次大小
- 改进网络架构（使用残差连接、自注意力机制等）

### Q3: 模式崩溃（Mode Collapse）？

WGAN-GP 已经很大程度上缓解了模式崩溃问题，但如果仍然出现：
- 检查梯度惩罚是否正确实现
- 增加数据多样性
- 使用 minibatch discrimination
- 尝试不同的网络架构

### Q4: 内存不足？

- 减少批次大小
- 降低图片分辨率
- 使用梯度累积
- 启用混合精度训练（AMP）

## 参考资料

### 论文

1. **WGAN**: Arjovsky, M., Chintala, S., & Bottou, L. (2017). "Wasserstein GAN". ICML 2017.
2. **WGAN-GP**: Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017). "Improved Training of Wasserstein GANs". NIPS 2017.
3. **GAN**: Goodfellow, I., Pouget-Abadie, J., Mirza, M., et al. (2014). "Generative Adversarial Networks". NIPS 2014.

### 学习资源

- [GAN 教程 - Ian Goodfellow (NIPS 2016)](https://arxiv.org/abs/1701.00160)
- [PyTorch WGAN-GP 实现](https://github.com/caogang/wgan-gp)
- [深度学习花书 - Goodfellow et al.](https://www.deeplearningbook.org/)

### 相关项目

- [StyleGAN](https://github.com/NVlabs/stylegan)
- [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch)
- [Progressive GAN](https://github.com/tkarras/progressive_growing_of_gans)

## 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交改动 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目采用 Apache License 2.0 许可证 - 详见 [LICENSE](LICENSE) 文件

## 致谢

- 感谢 Wasserstein GAN 和 WGAN-GP 的原作者
- 感谢人工智能原理与应用课程的老师和同学们
- 感谢开源社区提供的各种工具和资源

## 联系方式

- 项目仓库: [https://github.com/JY226622/WGAN-GP](https://github.com/JY226622/WGAN-GP)
- 问题反馈: 请在 GitHub Issues 中提出

---

**注意**: 本项目仅用于学习和研究目的，请勿用于商业用途。生成的二次元图片版权归原创作者所有。

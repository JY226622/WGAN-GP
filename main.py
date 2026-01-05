import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image
import glob
from generator import Generator
from critic import Critic
# ====================== 全局配置======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


batch_size = 128
lr = 0.0001
beta1 = 0.0
beta2 = 0.999
num_epochs = 50
latent_dim = 100
img_size = 96
channels = 3
sample_interval = 100
save_model_interval = 5
critic_iterations = 5  # WGAN-GP核心：Critic（判别器）训练5步，生成器训练1步
lambda_gp = 10  # 梯度惩罚系数

dataset_path = r'D:\AApython\Anime-WGAN-main\dataset'
output_path = './output'
model_path = './models'

os.makedirs(output_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)


transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# ====================== WGAN-GP核心：梯度惩罚计算函数 ======================
def compute_gradient_penalty(critic, real_imgs, fake_imgs, device):
    """
    计算WGAN-GP的梯度惩罚项
    :param critic: 判别器（Critic）
    :param real_imgs: 真实图片
    :param fake_imgs: 生成图片
    :param device: 训练设备（GPU/CPU）
    :return: 梯度惩罚损失
    """
    # 1. 随机采样插值系数ε（0~1均匀分布）
    batch_size = real_imgs.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device, requires_grad=True)
    # 2. 计算真实图片和生成图片的插值样本x_hat
    x_hat = epsilon * real_imgs + (1 - epsilon) * fake_imgs
    # 3. 计算Critic对x_hat的预测分数
    critic_x_hat = critic(x_hat)
    # 4. 计算预测分数对x_hat的梯度
    gradients = torch.autograd.grad(
        outputs=critic_x_hat,
        inputs=x_hat,
        grad_outputs=torch.ones_like(critic_x_hat),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    # 5. 调整梯度形状并计算梯度范数
    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = torch.norm(gradients, p=2, dim=1)
    # 6. 计算梯度惩罚损失：λ * (||∇x_hat C(x_hat)||_2 - 1)^2
    gradient_penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
    return gradient_penalty

# ====================== 模式1：WGAN-GP训练模型函数 ======================
def train_model():
    """WGAN-GP训练函数：执行完整的生成器和Critic训练流程"""
    # 加载数据集和数据加载器
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 初始化生成器和Critic（判别器）
    generator = Generator(latent_dim=latent_dim, img_size=img_size, channels=channels).to(device)
    critic = Critic(img_size=img_size, channels=channels).to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_D = optim.Adam(critic.parameters(), lr=lr, betas=(beta1, beta2))

    # 固定噪声
    fixed_noise = torch.randn(16, latent_dim, 1, 1).to(device)

    # 训练循环
    print(f"开始WGAN-GP训练，共 {num_epochs} 个Epoch，Critic训练步数：{critic_iterations}，使用设备：{device}")
    for epoch in range(num_epochs):
        loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=True)
        for i, (imgs, _) in loop:
            imgs = imgs.to(device)
            current_batch_size = imgs.size(0)

            # ---------------------
            #  训练Critic（判别器）：重复critic_iterations次
            # ---------------------
            for _ in range(critic_iterations):
                optimizer_D.zero_grad()

                # 生成假图片
                z = torch.randn(current_batch_size, latent_dim, 1, 1).to(device)
                fake_imgs = generator(z)

                # 计算Critic对真实图片和假图片的分数
                real_score = critic(imgs)
                fake_score = critic(fake_imgs.detach())

                # 计算Critic损失（WGAN损失：-E[C(x_real)] + E[C(x_fake)]）
                d_loss = -torch.mean(real_score) + torch.mean(fake_score)

                # 计算梯度惩罚并加入总损失
                gp = compute_gradient_penalty(critic, imgs, fake_imgs, device)
                d_loss_total = d_loss + gp

                # 反向传播更新Critic
                d_loss_total.backward()
                optimizer_D.step()

            # ---------------------
            #  训练生成器
            # ---------------------
            optimizer_G.zero_grad()

            # 生成假图片并计算Critic分数
            fake_imgs = generator(z)
            fake_score = critic(fake_imgs)

            # 生成器损失（WGAN损失：-E[C(G(z))]）
            g_loss = -torch.mean(fake_score)

            # 反向传播更新生成器
            g_loss.backward()
            optimizer_G.step()

            # 更新进度条
            loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
            loop.set_postfix(Critic_loss=d_loss.item(), G_loss=g_loss.item(), GP_loss=gp.item())

            # 保存生成图片
            if (i + 1) % sample_interval == 0:
                with torch.no_grad():
                    fake_imgs = generator(fixed_noise).detach().cpu()
                save_image(fake_imgs, os.path.join(output_path, f"epoch_{epoch + 1}_step_{i + 1}.png"),
                           nrow=4, normalize=True)

        # 保存模型
        if (epoch + 1) % save_model_interval == 0:
            torch.save(generator.state_dict(), os.path.join(model_path, f"generator_epoch_{epoch + 1}.pth"))
            torch.save(critic.state_dict(), os.path.join(model_path, f"critic_epoch_{epoch + 1}.pth"))  # 重命名为Critic

    # 保存最终模型
    torch.save(generator.state_dict(), os.path.join(model_path, "generator_final.pth"))
    torch.save(critic.state_dict(), os.path.join(model_path, "critic_final.pth"))
    print("WGAN-GP训练完成！最终模型已保存至 ./models 目录")

# ====================== 模式2：加载模型生成图片函数 =====================
def generate_anime_images(num_images=16, pretrained_model_path=None):
    """
    加载WGAN-GP预训练生成器模型，生成二次元图片
    """
    if pretrained_model_path is None:  # 如果用户没有指定预训练模型路径
        pretrained_model_path = os.path.join(model_path, "generator_final.pth")  # 使用默认路径

    if not os.path.exists(pretrained_model_path):  # 检查模型文件是否存在
        print(f"错误：未找到预训练模型文件 {pretrained_model_path}")  # 提示错误信息
        print("请先执行训练模式生成模型，或指定正确的模型路径")  # 提示用户解决方法
        return  # 退出函数

    # 初始化生成器
    gen = Generator(latent_dim=latent_dim, img_size=img_size, channels=channels).to(device)  # 创建生成器实例并移动到指定设备
    gen.load_state_dict(torch.load(pretrained_model_path, map_location=device))  # 加载预训练模型权重
    gen.eval()  # 将生成器设置为评估模式

    # 生成随机噪声
    z = torch.randn(num_images, latent_dim, 1, 1).to(device)  # 生成形状为(num_images, latent_dim, 1, 1)的随机噪声张量

    # 生成图片
    print(f"正在生成 {num_images} 张二次元图片，使用设备：{device}")  # 打印生成信息
    with torch.no_grad():  # 禁用梯度计算，加快推理速度
        fake_imgs = gen(z).detach().cpu()  # 生成图片并将结果从GPU移到CPU

    # 保存图片
    os.makedirs(output_path, exist_ok=True)  # 确保输出目录存在
    for i in range(num_images):
        img_path = os.path.join(output_path, f"generated_{i}.png")  # 每张图单独路径
        save_image(fake_imgs[i], img_path, normalize=True)  # 保存单张图片
        print(f"生成的图片已保存到：{img_path}")  # 打印保存信息

# ====================== 主程序：模式选择交互 =====================
if __name__ == '__main__':
    print("=" * 50)
    print("WGAN-GP 二次元头像生成工具")
    print("=" * 50)
    print("请选择运行模式：")
    print("1. 训练模式（训练生成器和Critic，需要数据集）")
    print("2. 生成模式（加载预训练模型，生成二次元图片）")
    print("=" * 50)

    # 获取用户输入
    while True:
        try:
            mode_choice = int(input("请输入模式编号（1/2）："))
            if mode_choice in [1, 2]:
                break
            else:
                print("输入无效！请仅输入 1 或 2")
        except ValueError:
            print("输入无效！请输入数字 1 或 2")

    # 执行对应模式
    if mode_choice == 1:
        train_model()
    else:
        try:
            num_imgs = int(input("请输入要生成的图片数量："))
            generate_anime_images(num_images=num_imgs)
        except ValueError:
            print("输入无效，使用默认数量 16 张")
            generate_anime_images(num_images=16)
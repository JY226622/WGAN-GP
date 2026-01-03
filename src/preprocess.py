"""
数据预处理脚本
将原始图像预处理为统一格式
"""
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image


def preprocess_images(input_dir, output_dir, image_size=64, format='png'):
    """
    预处理图像
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        image_size: 目标图像大小
        format: 输出格式
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 支持的图像格式
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    
    # 获取所有图像文件
    image_files = []
    for ext in valid_extensions:
        image_files.extend(Path(input_dir).rglob(f'*{ext}'))
        image_files.extend(Path(input_dir).rglob(f'*{ext.upper()}'))
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    if len(image_files) == 0:
        print(f"错误: 在 {input_dir} 中未找到图像文件")
        return
    
    # 处理每个图像
    processed_count = 0
    error_count = 0
    
    for img_path in tqdm(image_files, desc="处理图像"):
        try:
            # 打开图像
            img = Image.open(img_path)
            
            # 转换为RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 调整大小（保持纵横比，然后居中裁剪）
            # 1. 先缩放使最短边等于image_size
            width, height = img.size
            if width < height:
                new_width = image_size
                new_height = int(height * image_size / width)
            else:
                new_height = image_size
                new_width = int(width * image_size / height)
            
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # 2. 中心裁剪到 image_size x image_size
            left = (new_width - image_size) // 2
            top = (new_height - image_size) // 2
            right = left + image_size
            bottom = top + image_size
            
            img = img.crop((left, top, right, bottom))
            
            # 保存
            output_filename = f"processed_{processed_count:06d}.{format}"
            output_path = os.path.join(output_dir, output_filename)
            img.save(output_path, format.upper())
            
            processed_count += 1
            
        except Exception as e:
            print(f"错误处理 {img_path}: {e}")
            error_count += 1
    
    print(f"\n处理完成!")
    print(f"成功: {processed_count} 张")
    print(f"失败: {error_count} 张")
    print(f"输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess anime images')
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory containing raw images')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for processed images')
    parser.add_argument('--size', type=int, default=64,
                        help='Target image size (default: 64)')
    parser.add_argument('--format', type=str, default='png',
                        choices=['png', 'jpg', 'jpeg'],
                        help='Output image format (default: png)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("图像预处理工具")
    print("=" * 80)
    print(f"输入目录: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"目标大小: {args.size}x{args.size}")
    print(f"输出格式: {args.format}")
    print("=" * 80)
    
    preprocess_images(
        args.input,
        args.output,
        image_size=args.size,
        format=args.format
    )


if __name__ == "__main__":
    main()

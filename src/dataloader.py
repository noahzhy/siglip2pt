import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import pandas as pd
import os
import re
import random
import numpy as np

# ----------------------------------------------------------------
# 1. 文本增强与标准化工具 (核心部分)
# ----------------------------------------------------------------
class TextNormalizer:
    def __init__(self):
        # 预编译正则，匹配常见的容量单位：ml, l, kg, g, oz, lb, pack, pcs
        # 目标是捕获 "数字 + 空格(可选) + 单位" 的模式
        self.volume_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(ml|l|kg|gm|g|oz|lb|pcs|pack|pk)', re.IGNORECASE)
        
        # 定义一些口味关键词（根据你的实际业务扩展）
        # 如果文本中包含这些词，我们可以选择将其“加粗”或前置
        self.flavor_keywords = ["spicy", "sweet", "sour", "chicken", "beef", "vanilla", "chocolate", "strawberry"]

    def normalize_volume(self, text):
        """
        解决 Volume Confuse: 
        1. 统一单位格式 (例如 1.5 L -> 1.5l)。
        2. 去除数字和单位间的空格，让 Tokenizer 产生更紧凑的特征。
        """
        def replace_func(match):
            number = match.group(1)
            unit = match.group(2).lower()
            # 统一单位映射
            if unit == 'gm': unit = 'g'
            if unit == 'pk': unit = 'pack'
            return f"{number}{unit}" # 强制去除空格，如 "500 ml" -> "500ml"

        return self.volume_pattern.sub(replace_func, text)

    def highlight_attributes(self, text):
        """
        简单的文本增强：
        随机打乱非关键部分的语序，或者强调关键属性，迫使模型关注 Volume 和 Flavor
        """
        # 这里演示一个简单的逻辑：确保容量在文本中清晰可见
        # 实际训练中，可以随机丢弃一些无意义的词 (stop words)
        return text.strip()

    def __call__(self, text):
        text = self.normalize_volume(str(text))
        text = self.highlight_attributes(text)
        return text

# ----------------------------------------------------------------
# 2. 图像增强定义
# ----------------------------------------------------------------
def get_train_transforms(image_size=224):
    return transforms.Compose([
        # 随机缩放裁剪：让模型学会看局部（比如只看瓶底的容量，或只看口味图）
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        # 随机水平翻转
        transforms.RandomHorizontalFlip(p=0.5),
        # 随机旋转 90 度 或者 270 度
        transforms.RandomRotation(degrees=[90, 270]),
        # 颜色抖动：模拟不同光照，非常重要
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        # DO NOT convert to Tensor/Normalize, SiglipProcessor will do that
        # output is PIL Image
    ])

# ----------------------------------------------------------------
# 3. Dataset 定义
# ----------------------------------------------------------------
class SiglipDataset(Dataset):
    def __init__(self, csv_file, image_root_dir, processor, transform=None):
        """
        Args:
            csv_file: 包含 'image_path' 和 'caption' 列的 CSV 文件路径
            image_root_dir: 图片文件夹根目录
            processor: Hugging Face 的 SiglipProcessor
            transform: torchvision transforms
        """
        self.data = pd.read_csv(csv_file)
        self.image_root_dir = image_root_dir
        self.processor = processor
        self.transform = transform
        self.text_normalizer = TextNormalizer()

        # 简单的校验
        if 'image_path' not in self.data.columns or 'caption' not in self.data.columns:
            raise ValueError("CSV must contain 'image_path' and 'caption' columns")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # 1. 加载图片
        img_path = os.path.join(self.image_root_dir, row['image_path'])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Error loading image {img_path}: {e}")
            # 返回一个黑图防止崩掉，或者随机取一张
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        # 2. 图像增强
        if self.transform:
            image = self.transform(image)

        # 3. 文本处理
        raw_caption = row['caption']
        
        # 应用我们定义的标准化逻辑，处理 Volume 和 Flavor
        clean_caption = self.text_normalizer(raw_caption)

        # 4. 返回原始数据
        # 注意：我们这里不直接返回 tokenized tensor，而是返回 raw data
        # 让 Collate Function 统一做 Padding，效率更高
        return {
            "image": image,
            "text": clean_caption
        }

# ----------------------------------------------------------------
# 4. Collate Function (批处理函数)
# ----------------------------------------------------------------
class SiglipCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        images = [item['image'] for item in batch]
        texts = [item['text'] for item in batch]

        # 调用 Processor 进行批量处理
        # return_tensors="pt" 会自动转换为 PyTorch Tensor
        # padding=True 会自动 pad 到当前 batch 中最长的文本
        inputs = self.processor(
            text=texts,
            images=images,
            padding="max_length", # 建议训练时设为 max_length 以保证维度一致
            max_length=256,        # 根据你的文本平均长度调整，太长浪费显存
            truncation=True,
            return_tensors="pt"
        )
        return inputs

# ----------------------------------------------------------------
# 5. 使用示例 (如果你直接运行此脚本)
# ----------------------------------------------------------------
if __name__ == "__main__":
    from transformers import SiglipProcessor

    MODEL_ID = "google/siglip2-base-patch16-224"
    processor = SiglipProcessor.from_pretrained(MODEL_ID)

    dummy_data = {
        'image_path': ['img1.jpg', 'img2.jpg'],
        'caption': ['Delicious spicy beef noodles 500 ml pack', 'Sweet Vanilla Ice Cream 1.5kg']
    }
    df = pd.DataFrame(dummy_data)
    df.to_csv('dummy_train.csv', index=False)

    Image.new('RGB', (100, 100)).save('img1.jpg')
    Image.new('RGB', (100, 100)).save('img2.jpg')

    dataset = SiglipDataset(
        csv_file='dummy_train.csv', 
        image_root_dir='.', 
        processor=processor,
        transform=get_train_transforms()
    )

    collator = SiglipCollator(processor)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collator)

    # 测试读取
    for batch in dataloader:
        print("Batch keys:", batch.keys())
        print("Image shape:", batch['pixel_values'].shape)
        print("Input IDs shape:", batch['input_ids'].shape)
        
        # 验证文本清洗效果
        # 需要 decode 出来看是否变成了 500ml (无空格)
        decoded_text = processor.batch_decode(batch['input_ids'], skip_special_tokens=True)
        print("Processed Texts:", decoded_text)
        break

    os.remove('dummy_train.csv')
    os.remove('img1.jpg')
    os.remove('img2.jpg')

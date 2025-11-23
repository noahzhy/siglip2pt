import torch
from PIL import Image
from transformers import (
    SiglipProcessor, 
    SiglipForImageAndTextRetrieval, 
    Trainer, 
    TrainingArguments
)
from datasets import Dataset

# ----------------------------------------------------------------
# 1. 配置与模型加载
# ----------------------------------------------------------------
# 请替换为实际的 SigLIP 2 模型 ID，例如 "google/siglip2-so400m-patch14-384"
# 如果 SigLIP 2 尚未在 HF 完全索引，可使用兼容的 SigLIP 1 权重测试，如 "google/siglip-so400m-patch14-384"
MODEL_ID = "google/siglip-so400m-patch14-384" 
OUTPUT_DIR = "./siglip2_finetuned"

# 加载处理器和模型
processor = SiglipProcessor.from_pretrained(MODEL_ID)
model = SiglipForImageAndTextRetrieval.from_pretrained(MODEL_ID)

# ----------------------------------------------------------------
# 2. 准备数据 (这里使用虚拟数据作为示例)
# ----------------------------------------------------------------
def create_dummy_dataset():
    # 创建一些随机 RGB 图像和文本
    data = []
    for i in range(20): # 示例 20 条数据
        img = Image.new('RGB', (384, 384), color=(i*10, 100, 200))
        text = f"This is a caption for image number {i}"
        data.append({"image": img, "text": text})
    return Dataset.from_list(data)

dataset = create_dummy_dataset()

# ----------------------------------------------------------------
# 3. 数据整理函数 (Data Collator)
# ----------------------------------------------------------------
def collate_fn(batch):
    images = [x['image'] for x in batch]
    texts = [x['text'] for x in batch]
    
    # Processor 负责同时处理图像和文本
    inputs = processor(
        text=texts,
        images=images,
        padding="max_length", # 补全到最大长度
        truncation=True,
        max_length=64,        # 根据实际文本长度调整
        return_tensors="pt"
    )
    
    return inputs

# ----------------------------------------------------------------
# 4. 训练参数配置
# ----------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4, # SigLIP 对 Batch Size 不像 CLIP 那么敏感，但显存允许越大越好
    learning_rate=1e-5,            # 微调建议低学习率
    num_train_epochs=3,
    logging_steps=5,
    save_steps=50,
    remove_unused_columns=False,   # 关键：防止 Trainer 移除 dataset 中的 image 列
    fp16=torch.cuda.is_available(), # 开启混合精度
    report_to="none"
)

# ----------------------------------------------------------------
# 5. 开始训练
# ----------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn,
)

print("开始训练 SigLIP 2...")
trainer.train()

# 保存模型和处理器
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"训练完成，模型已保存至 {OUTPUT_DIR}")
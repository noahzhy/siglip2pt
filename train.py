import torch
import torch.nn as nn
from PIL import Image
from transformers import SiglipProcessor, SiglipModel, Trainer, TrainingArguments
from datasets import Dataset

# ----------------------------------------------------------------
# 1. 定义模型包装类 (关键步骤)
# ----------------------------------------------------------------
class SiglipForFineTuning(nn.Module):
    """
    包装 HuggingFace 的 SiglipModel，以便在 forward 中计算 Loss。
    """
    def __init__(self, model_id):
        super().__init__()
        self.model = SiglipModel.from_pretrained(model_id)
        # SigLIP 的核心是用 Sigmoid 计算 Loss，而不是 Softmax
        # 这里使用 BCEWithLogitsLoss，目标矩阵是单位矩阵（对角线为1，其余为0）
        self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, pixel_values, attention_mask=None, **kwargs):
        # 1. 获取模型输出
        outputs = self.model(
            input_ids=input_ids, 
            pixel_values=pixel_values, 
            attention_mask=attention_mask
        )

        # logits_per_image 形状为 (Batch_Size, Batch_Size)
        # 已经包含了 SigLIP 特有的 bias 和 temperature 缩放
        logits = outputs.logits_per_image
        
        # 2. 构建标签 (Labels)
        # 对角线是正样本 (1.0)，其他位置是负样本 (0.0)
        batch_size = logits.shape[0]
        labels = torch.eye(batch_size, device=logits.device)
        
        # 3. 计算 Loss
        loss = self.loss_fct(logits, labels)
        
        # 4. 必须返回字典，且包含 'loss' 键，Trainer 才能工作
        return {"loss": loss, "logits": logits}

    # 让 Trainer 能够保存内部的 HF 模型
    def save_pretrained(self, save_directory):
        self.model.save_pretrained(save_directory)

# ----------------------------------------------------------------
# 2. 配置与加载
# ----------------------------------------------------------------
# MODEL_ID = "google/siglip-so400m-patch14-384" # 或者 siglip2 对应的 checkpoint
MODEL_ID = "google/siglip2-base-patch16-224"
processor = SiglipProcessor.from_pretrained(MODEL_ID)

# 实例化我们自定义的模型
model = SiglipForFineTuning(MODEL_ID)

# ----------------------------------------------------------------
# 3. 准备虚拟数据
# ----------------------------------------------------------------
def create_dummy_dataset():
    data = []
    for i in range(32): # 创建 32 条样本
        # 随机颜色图片
        img = Image.new('RGB', (384, 384), color=(i*5 % 255, (i*10)%255, 150))
        text = f"This is a photo of color id {i}"
        data.append({"image": img, "text": text})
    return Dataset.from_list(data)

dataset = create_dummy_dataset()

# ----------------------------------------------------------------
# 4. 数据整理 (Collate Function)
# ----------------------------------------------------------------
def collate_fn(batch):
    images = [x['image'] for x in batch]
    texts = [x['text'] for x in batch]
    
    inputs = processor(
        text=texts,
        images=images,
        padding="max_length",
        truncation=True,
        max_length=32,
        return_tensors="pt"
    )
    return inputs

# ----------------------------------------------------------------
# 5. 训练参数
# ----------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./siglip_output",
    per_device_train_batch_size=4,  # 显存允许的话越大越好
    num_train_epochs=3,
    learning_rate=5e-6,             # SigLIP 也是微调，学习率要低
    logging_steps=1,
    remove_unused_columns=False,    # 必须设为 False，否则 image 字段会被过滤掉
    save_strategy="no",
    report_to="none",
    fp16=torch.cuda.is_available()  # 推荐开启混合精度
)

# ----------------------------------------------------------------
# 6. 开始训练
# ----------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn,
)

print("开始训练...")
trainer.train()

# 保存最终模型
model.save_pretrained("./siglip_output/final_model")
processor.save_pretrained("./siglip_output/final_model")
print("训练完成。")
import torch
from PIL import Image
from transformers import SiglipProcessor, SiglipModel
import torch.nn.functional as F
import numpy as np

# ----------------------------------------------------------------
# 1. 配置路径与设备
# ----------------------------------------------------------------
# 这里可以是 HuggingFace ID，也可以是你上一轮训练保存的本地路径
MODEL_PATH = "./siglip_output/final_model" 

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"正在加载模型: {MODEL_PATH} (使用设备: {device})...")

# ----------------------------------------------------------------
# 2. 加载模型和处理器
# ----------------------------------------------------------------
# 注意：我们可以直接加载 SiglipModel，它包含了 Vision Encoder 和 Projection Head
model = SiglipModel.from_pretrained(MODEL_PATH).to(device)
processor = SiglipProcessor.from_pretrained(MODEL_PATH)

# 设为评估模式，关闭 Dropout 等
model.eval()

# ----------------------------------------------------------------
# 3. 定义提取函数
# ----------------------------------------------------------------
def get_image_embeddings(image_paths, normalize=True):
    """
    输入图片路径列表，输出 Numpy 格式的 Embeddings
    """
    images = []
    for path in image_paths:
        # 打开图片并转为 RGB (防止 PNG 透明通道报错)
        images.append(Image.open(path).convert("RGB"))

    # 预处理：Resize, Normalize, 转 Tensor
    inputs = processor(images=images, return_tensors="pt").to(device)

    # 推理
    with torch.no_grad():
        # get_image_features 会自动运行 Vision Model 并通过 Projection 层
        image_features = model.get_image_features(**inputs)

    # 归一化 (可选但推荐)
    # SigLIP/CLIP 的特征通常需要归一化，这样点积 (Dot Product) 就等于余弦相似度
    if normalize:
        image_features = F.normalize(image_features, p=2, dim=-1)

    # 转回 CPU 并转为 numpy 数组，方便存入向量数据库 (如 Milvus, Faiss)
    return image_features.cpu().numpy()

# ----------------------------------------------------------------
# 4. 运行示例
# ----------------------------------------------------------------
if __name__ == "__main__":
    img_paths = ["coke.jpg", "coke.jpg"]

    print("开始提取特征...")
    embeddings = get_image_embeddings(img_paths)
    
    print(f"特征提取成功，形状: {embeddings.shape}")
    # SigLIP SO400M 的输出维度通常是 1152
    
    # 简单计算一下两张图的相似度
    # 因为已经归一化了，直接矩阵相乘就是余弦相似度
    similarity = embeddings[0] @ embeddings[1].T
    print(f"图片 1 和 图片 2 的相似度: {similarity:.4f}")

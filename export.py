import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SiglipModel, SiglipProcessor
import onnx

# ----------------------------------------------------------------
# 1. 配置
# ----------------------------------------------------------------
MODEL_PATH = "./siglip_output/final_model" # 或者你微调后的路径
ONNX_PATH = "siglip_vision.onnx"
OPSET_VERSION = 18  # 推荐使用 14 或更高，支持更多算子

# ----------------------------------------------------------------
# 2. 定义包装类 (Wrapper)
# ----------------------------------------------------------------
# 这一步非常重要：
# 1. 剥离 HF 的复杂输出，只返回 Tensor。
# 2. 将 L2 归一化内嵌到模型中。
class SiglipVisionExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        # 调用 SigLIP 提取特征
        # get_image_features 包含了 VisionTransformer + Projection Head
        features = self.model.get_image_features(pixel_values)
        
        # 内置 L2 归一化
        # 这样部署时，直接拿出来的向量就是归一化过的
        return F.normalize(features, p=2, dim=-1)

# ----------------------------------------------------------------
# 3. 加载模型
# ----------------------------------------------------------------
print(f"正在加载 PyTorch 模型: {MODEL_PATH}...")
hf_model = SiglipModel.from_pretrained(MODEL_PATH)
hf_model.eval()

# 包装模型
onnx_model = SiglipVisionExport(hf_model)

# ----------------------------------------------------------------
# 4. 准备 Dummy Input (虚拟输入)
# ----------------------------------------------------------------
# SigLIP Base 输入分辨率通常是 224x224
# SigLIP SO400M 输入分辨率通常是 384x384
# 格式: [Batch_Size, Channels, Height, Width]
dummy_input = torch.randn(8, 3, 224, 224)

# ----------------------------------------------------------------
# 5. 执行导出
# ----------------------------------------------------------------
print(f"正在导出为 ONNX (Opset {OPSET_VERSION})...")

torch.onnx.export(
    onnx_model,                 # 包装后的模型
    dummy_input,                # 虚拟输入
    ONNX_PATH,                  # 输出路径
    export_params=True,         # 存储权重
    opset_version=OPSET_VERSION,
    do_constant_folding=True,   # 优化常量
    input_names=['pixel_values'],
    output_names=['image_embeds'],
    # 动态轴：允许推理时 Batch Size 可变
    dynamic_axes={
        'pixel_values': {0: 'batch_size'},
        'image_embeds': {0: 'batch_size'}
    }
)

print(f"导出成功！模型已保存至: {ONNX_PATH}")

# ----------------------------------------------------------------
# 6. 验证 ONNX 模型 (可选)
# ----------------------------------------------------------------
try:
    import onnxruntime as ort
    import numpy as np
    
    print("正在验证 ONNX 模型...")
    session = ort.InferenceSession(ONNX_PATH)
    
    # 构造 numpy 输入
    ort_input = {session.get_inputs()[0].name: dummy_input.numpy()}
    
    # 推理
    ort_output = session.run(None, ort_input)[0]
    
    print(f"验证成功！输出形状: {ort_output.shape}")
    print(f"前5个数值: {ort_output[0][:5]}")
    
except ImportError:
    print("未安装 onnxruntime，跳过验证。")

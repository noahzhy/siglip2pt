import onnx
from onnxsim import simplify
import os

# ----------------------------------------------------------------
# 配置
# ----------------------------------------------------------------
INPUT_PATH = "siglip_vision.onnx"       # 原始导出的模型
OUTPUT_PATH = "siglip_vision_sim.onnx"  # 简化后的模型

def simplify_model():
    print(f"正在读取模型: {INPUT_PATH} ...")
    
    # 1. 加载 ONNX 模型
    try:
        model = onnx.load(INPUT_PATH)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {INPUT_PATH}")
        return

    # 2. 执行简化
    # simplify 返回两个值：简化后的模型对象 和 验证状态(bool)
    # dynamic_input_shape=True 可以在某些复杂动态尺寸情况下更安全，
    # 但通常默认参数对于 SigLIP 这种标准 Vision Transformer 已经足够。
    print("正在运行 onnxsim 进行图优化...")
    model_simp, check = simplify(model)

    # 3. 验证
    if not check:
        print("警告: onnxsim 验证失败！生成的模型可能不可用。")
    else:
        print("onnxsim 验证通过。")

    # 4. 保存
    onnx.save(model_simp, OUTPUT_PATH)
    
    # 5. 对比文件大小
    original_size = os.path.getsize(INPUT_PATH) / (1024 * 1024)
    sim_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    
    print("-" * 30)
    print(f"原始模型大小: {original_size:.2f} MB")
    print(f"简化模型大小: {sim_size:.2f} MB")
    print(f"模型已保存至: {OUTPUT_PATH}")
    print("-" * 30)

if __name__ == "__main__":
    simplify_model()

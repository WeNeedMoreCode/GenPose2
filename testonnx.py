import onnx

# 替换为你的ONNX文件绝对路径
onnx_path = "/home/syx/ModelZoo-PyTorch/ACL_PyTorch/built-in/embodied_ai/GenPose22/GenPose2/onnx_models/score_net.onnx"
model = onnx.load(onnx_path)

print("===== ONNX模型真实输入信息 =====")
for idx, inp in enumerate(model.graph.input):
    # 打印输入名（重点）+ 形状
    name = inp.name
    shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim if dim.dim_value != 0]
    print(f"输入{idx+1}：名={name}，形状={shape}")
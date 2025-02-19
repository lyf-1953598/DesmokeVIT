import torch

# 加载模型（假设模型文件名为model.pth）
model_path = "checkpoints/vit2_new_200epoch_vgg/latest_net_G.pth"  # 这里改为实际的文件路径
model = torch.load(model_path)

# 如果模型是保存的state_dict，则加载state_dict
# model = MyModel()  # 如果有需要的话，可以先实例化模型
# model.load_state_dict(torch.load(model_path))

# 计算模型的参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

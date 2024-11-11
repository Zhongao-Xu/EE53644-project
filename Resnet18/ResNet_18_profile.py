import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch.profiler import profile, record_function, ProfilerActivity

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 使用预训练的 ResNet18 模型并加载保存的参数
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 10)  # 修改最后一层以适应 CIFAR-10 数据集
model.load_state_dict(torch.load('resnet18_cifar10.pth'))  # 加载保存的模型参数
model = model.to(device)

# 设置为评估模式
model.eval()

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将 CIFAR-10 图像大小调整为 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载 CIFAR-10 测试集
test_data = CIFAR10(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

# 模型性能分析
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True
) as prof:
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            with record_function("model_inference"):
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

# 打印分析结果
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

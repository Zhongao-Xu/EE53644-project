import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


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

# 模型预测
correct = 0
total = 0

with torch.no_grad():  # 通过不计算梯度加速推理
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# 打印测试集上的准确率
accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

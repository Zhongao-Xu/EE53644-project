import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torch.nn as nn
import torch.optim as optim

# 检查 CUDA 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练的 ResNet-18 模型
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# 调整最后的全连接层，因为 CIFAR-10 只有 10 个类别
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整 CIFAR-10 图像大小到 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
])

# 加载 CIFAR-10 数据集
train_data = CIFAR10(root='./data', train=True, transform=transform, download=True)
test_data = CIFAR10(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练模型
num_epochs = 1
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印训练进度
        running_loss += loss.item()
        if (i + 1) % 100 == 0:  # 每100批次打印一次
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1}, Average Loss: {running_loss / len(train_loader):.4f}")
torch.save(model.state_dict(), 'Resnet18_cifar10.pth') # Save model parameters

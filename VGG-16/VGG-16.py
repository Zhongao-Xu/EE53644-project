import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# Loading pre-trained VGG-16 models
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
model.classifier[6] = torch.nn.Linear(4096, 10)
model = model.to('cuda')

# Defining Data Conversions
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize CIFAR-10 image to 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Loading the CIFAR-10 dataset
train_data = CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)  # Setting the batch size, Change this depend on CPU and GPU 8/16/32.

# Defining the loss function and optimiser
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# training cycle
model.train()
for epoch in range(1):  # 1 rounds of training
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to('cuda'), labels.to('cuda')

        # forward propagation
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backpropagation and optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 100 == 0:  # Printed every 100 batches
            print(f"Epoch [{epoch+1}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1}, Average Loss: {running_loss / len(train_loader):.4f}")
torch.save(model.state_dict(), 'vgg16_cifar10.pth') # Save model parameters
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# Load VGG-16 model and modify output layer
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
model.classifier[6] = torch.nn.Linear(4096, 10)  # Ensure that the output layer is consistent with the training
model.load_state_dict(torch.load('vgg16_cifar10.pth'))  # Load saved model parameters
model = model.to('cuda')

# Set to evaluation mode
model.eval()

# Defining Data Conversions
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Loading the CIFAR-10 test set
test_data = CIFAR10(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

# model prediction
correct = 0
total = 0

with torch.no_grad():  # Accelerated inference by not calculating gradients
    for images, labels in test_loader:
        images, labels = images.to('cuda'), labels.to('cuda')

        # forward propagation
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Printing Accuracy on Test Sets
accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

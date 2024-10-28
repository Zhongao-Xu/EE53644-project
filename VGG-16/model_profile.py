import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch.profiler import profile, record_function, ProfilerActivity

# Load and modify VGG-16 model
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
model.classifier[6] = torch.nn.Linear(4096, 10)
model.load_state_dict(torch.load('vgg16_cifar10.pth'))
model = model.to('cuda')
model.eval()

# Define Data Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 Test Set
test_data = CIFAR10(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

# Profiling
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            with record_function("model_inference"):
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

# Print profiler results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

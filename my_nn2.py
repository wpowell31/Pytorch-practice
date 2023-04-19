"""Implementing Custom CNN."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


# MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root="/Users/willpowell/Desktop",
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)

test_dataset = torchvision.datasets.MNIST(
    root="/Users/willpowell/Desktop",
    train=False, transform=transforms.ToTensor()
)

# Data loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=100, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=100, shuffle=False
)

# Hyperparameters
num_epochs = 100
learning_rate = 0.03
num_classes = 10


class MNIST_CNN2(nn.Module):
    """Make custom MNIST_CNN."""

    def __init__(self):
        """Initialize MNIST_CNN2."""
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(14 * 14 * 64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        """Feed forward."""
        # conv layer 1
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        # conv layer 2
        x = self.conv2(x)
        x = F.relu(x)

        # conv layer 3
        x = self.conv3(x)
        x = F.relu(x)

        # fc layer 1
        x = x.view(-1, 14 * 14 * 128)
        x = self.fc1(x)
        x = F.relu(x)

        # fc layer 2
        x = self.fc2(x)
        return x


model = MNIST_CNN2()


# Initialize the loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the data
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Test the data
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(test_loader):
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (labels == predicted).sum().item()
    print(
        "Test Accuracy of the model on the 10000 test images: {} %".format(
            100 * correct / total
        )
    )

# Save the model checkpoint
torch.save(model.state_dict(), "model.ckpt")

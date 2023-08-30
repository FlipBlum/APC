import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(32 * 20 * 20, 128)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = nn.MaxPool2d(2)(x)
        x = nn.Dropout(0.25)(x)
        
        x = self.relu2(self.conv2(x))
        x = nn.MaxPool2d(2)(x)
        x = nn.Dropout(0.25)(x)
        
        x = self.relu3(self.conv3(x))
        x = nn.MaxPool2d(2)(x)
        x = nn.Dropout(0.25)(x)
        
        x = x.view(-1, 32 * 20 * 20)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

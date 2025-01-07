import torch
import torch.nn as nn
import torch.nn.functional as F

class AnimalFaceCNN(nn.Module):
    def __init__(self, num_classes):
        super(AnimalFaceCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # 입력 채널 3 (RGB), 출력 채널 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 풀링으로 이미지 크기 절반 축소
        self.fc1 = nn.Linear(128 * 64 * 64, 512)  # 512x512 -> 64x64 크기로 축소된 피처맵
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)  # 과적합 방지

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

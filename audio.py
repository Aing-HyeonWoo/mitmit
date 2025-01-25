import os
import torch
import torchaudio
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import torch.nn as nn

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model definition
class ConvBlock(nn.Module):
    def __init__(self, in_channel=1, out_channel=64, kernel_size=3, stride=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.layers(x)

class ResBlock(nn.Module):
    def __init__(self, in_channel, kernel_size=5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channel, in_channel, kernel_size=kernel_size, stride=1, padding='same'),
            nn.BatchNorm1d(in_channel),
            nn.ReLU(),
            nn.Conv1d(in_channel, in_channel, kernel_size=kernel_size, stride=1, padding='same'),
            nn.BatchNorm1d(in_channel)
        )

    def forward(self, x):
        return self.layers(x) + x  # Residual connection

class nEMGNet(nn.Module):
    def __init__(self, num_classes, n_channel_list=[64, 128, 256, 512], n_repeat=2):
        super().__init__()
        layers = []
        in_channels = 1
        for out_channels in n_channel_list:
            layers.append(ConvBlock(in_channel=in_channels, out_channel=out_channels))
            layers.append(nn.Sequential(*[ResBlock(out_channels) for _ in range(n_repeat)]))
            in_channels = out_channels

        self.cnn = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_channel_list[-1] * (22050 // 2**len(n_channel_list)), 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x

# 클래스 정의 (train.py에서 사용된 클래스와 동일해야 함)
classes = ['Cat', 'Chicken', 'Cow', 'Dog', 'Frog', 'Horse', 'Monkey', 'Sheep']
label_encoder = LabelEncoder()
label_encoder.fit(classes)

# 모델 초기화 및 로드
model_path = "./sound_model_FeedBack.pth"
num_classes = len(classes)
model = nEMGNet(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 데이터 전처리 함수
def preprocess_audio(file_path, target_length=22050):
    waveform, sample_rate = torchaudio.load(file_path)
    if waveform.size(0) == 2:
        waveform = waveform[0:1, :]
    if waveform.size(1) < target_length:
        padding = target_length - waveform.size(1)
        waveform = torch.cat([waveform, torch.zeros(1, padding)], dim=1)
    elif waveform.size(1) > target_length:
        waveform = waveform[:, :target_length]
    return waveform

def return_audio_label(file_path):
    waveform = preprocess_audio(file_path).to(device)
    waveform = waveform.unsqueeze(0).float()  # (1, 1, target_length)

# 예측 수행
    with torch.no_grad():
        outputs = model(waveform)
        _, predicted = torch.max(outputs, 1)
        predicted_label = label_encoder.inverse_transform([predicted.item()])[0]

    return predicted_label


# 결과 출력


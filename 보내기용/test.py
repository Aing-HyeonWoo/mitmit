# %%
import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50

# 클래스 맵
class_map = {"buffalo": 0, "cat": 1, "dog": 2, "elephant": 3, "rhino": 4, "zebra": 5}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드
model = resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 6)  # 클래스 개수
model.to(device)
model.load_state_dict(torch.load("./resnet50.pt"))

# 이미지 전처리 함수
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ResNet 기본 값
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # 배치 차원 추가

# 예측 함수
def pred(img):
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
    return predicted.item(), outputs  # 예측 값과 출력값 함께 반환

# 이미지 파일 경로와 레이블을 얻는 함수
def get_image(base_dir):
    image_paths = []
    true_labels = []
    for class_name, label in class_map.items():
        folder_path = os.path.join(base_dir, class_name)
        for img_file in os.listdir(folder_path):
            if img_file.endswith((".jpg", ".png", ".jpeg")):  # 이미지 파일만 필터링
                image_paths.append(os.path.join(folder_path, img_file))
                true_labels.append(label)
    return image_paths, true_labels

# 정확도를 계산하는 부분
correct_predictions = {label: 0 for label in class_map.values()}  # 클래스별 정확도 카운팅
total_predictions = {label: 0 for label in class_map.values()}  # 각 클래스에 대한 예측된 이미지 수 카운팅

# 데이터셋 준비
base_dir = "afhq/val"
img_path, true_labels = get_image(base_dir=base_dir)

# 각 클래스별로 100개씩 예측 진행
for idx, (image_path, true_label) in enumerate(zip(img_path, true_labels)):
    img_tensor = preprocess_image(image_path).to(device)  # GPU로 이동
    predicted_label, outputs = pred(img_tensor)

    # 출력값 확인 (추가)
    print(f"Image: {os.path.basename(image_path)}")
    print(f"True Label: {true_label}")
    print(f"Predicted Label: {predicted_label}")
    print(f"Model Outputs: {outputs}")
    print("-" * 30)

    # 예측이 맞은 경우 카운트
    if predicted_label == true_label:
        correct_predictions[true_label] += 1
    total_predictions[true_label] += 1

    # 100개 이상 예측을 진행했다면 종료
    if idx >= 99:
        break

# 각 클래스별 정확도 출력
for class_name, label in class_map.items():
    if total_predictions[label] > 0:
        accuracy = correct_predictions[label] / total_predictions[label] * 100
    else:
        accuracy = 0  # 예측한 이미지가 없는 경우 정확도 0%
        
    print(f"Class: {class_name}, Accuracy: {accuracy:.2f}%")
print()
# %%

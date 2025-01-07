# %%
import os
from PIL import Image
import torch
from torchvision import transforms
from model import AnimalFaceCNN  # 모델 클래스 임포트

# 1. 모델 불러오기
def load_model(model_path="animal_face_model.pth"):
    model = AnimalFaceCNN(num_classes=3)  # 학습 시 사용한 클래스 개수와 동일하게 설정
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 평가 모드로 설정
    return model

# 2. 이미지 전처리
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert('RGB')  # 이미지 열기 및 RGB로 변환
    return transform(image).unsqueeze(0)  # 배치 차원 추가

# 3. 예측 함수
def predict(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# 4. 데이터 가져오기
def get_image_paths_and_labels(base_dir):
    class_map = {"wild": 2}  # 클래스와 인덱스 매핑
    image_paths = []
    true_labels = []
    for class_name, label in class_map.items():
        folder_path = os.path.join(base_dir, class_name)
        for img_file in os.listdir(folder_path):
            if img_file.endswith((".jpg", ".png", ".jpeg")):  # 이미지 파일 필터링
                image_paths.append(os.path.join(folder_path, img_file))
                true_labels.append(label)
    return image_paths, true_labels

if __name__ == "__main__":
    # 폴더 경로 설정
    base_dir = "afhq/val"

    # 모델 로드
    model = load_model()

    # 이미지와 라벨 불러오기
    image_paths, true_labels = get_image_paths_and_labels(base_dir)

    # 클래스 매핑
    class_map = {0: "cat", 1: "dog"}

    # 예측 및 출력
    for idx, (image_path, true_label) in enumerate(zip(image_paths, true_labels)):
        image_tensor = preprocess_image(image_path)
        predicted_label = predict(model, image_tensor)

        # 파일 이름, 진짜 라벨, 예측 라벨 출력
        print(f"Image: {os.path.basename(image_path)}")
        print(f"True Label: {class_map[true_label]}")
        print(f"Predicted Label: {class_map[predicted_label]}")
        print("-" * 30)

        # 샘플 몇 개만 확인
        if idx >= 9:  # 상위 10개만 출력
            break

# %%

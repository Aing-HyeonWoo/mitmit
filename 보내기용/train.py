# %%
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.optim import lr_scheduler
from dataloader import dataset_loader
from dataloader import dataset_sizes
import copy
# https://2021-01-06getstarted.tistory.com/49 <- 감사한분

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 마지막 채널만 클래스 개수에 맞게 바꿔보아용
model = resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 6) # class 개수
model.to(device)
# %%
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=0.0001)

# %%

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 7, gamma= 0.001)

import torch
import torchvision.models as models



# ct = 0
# for child in model.children():
#     ct += 1
#     if ct < 6:
#         for param in child.parameters():
#             param.requires_grad = False



# %%
def train_resnet(model, criterion, optimizer, scheduler, num_epochs):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # 학습 모드
            else:
                model.eval()  # 평가 모드

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataset_loader[phase]:
                # 데이터를 GPU로 전송
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 옵티마이저 초기화
                optimizer.zero_grad()

                # 학습/평가 단계 설정
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    
                    # 출력 및 손실 확인

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 손실 확인

                    # 학습 단계에서만 역전파 수행
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # 손실 및 정확도 계산
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # 학습률 스케줄러 업데이트
            if phase == "train":
                scheduler.step()

            # 에포크 손실 및 정확도 계산
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # 최고 성능 모델 저장
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    print(f"Best val Acc: {best_acc:.4f}")

    # 최적의 가중치 로드
    model.load_state_dict(best_model_wts)
    return model

# 모델 학습 실행
model_resnet50 = train_resnet(model, criterion, optimizer, exp_lr_scheduler, num_epochs=3)

# 모델 저장
torch.save(model_resnet50.state_dict(), "resnet50.pt")


# %%

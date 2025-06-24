import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import os
import copy

# --- 1. 모델 아키텍처 및 하이퍼파라미터 정의 ---
# 이 값들은 실제 데이터셋과 실험 목적에 맞게 조정될 수 있습니다.

# 데이터 관련 파라미터
MAX_SEQ_LENGTH = 32
NUM_SINGERS = 100
NUM_SAMPLES = 100000  # 10만개 샘플
BATCH_SIZE = 256

# 모델 구조 관련 파라미터
SID_EMBEDDING_DIM = 16
GRU_UNITS = 128
NUM_GRU_LAYERS = 2
DROPOUT_RATE = 0.3

# 훈련 관련 파라미터
LEARNING_RATE = 0.001
NUM_EPOCHS = 50       # 최대 훈련 에포크 수
EARLY_STOPPING_PATIENCE = 5 # 검증 성능이 5 에포크 동안 개선되지 않으면 조기 종료

class DurationPredictorGRU(nn.Module):
    """
    가수 ID(SID)와 음표 길이 시퀀스를 입력받아,
    표현력 있는(expressive) 음표 길이 시퀀스를 예측하는 양방향 GRU 모델입니다.
    """
    def __init__(self):
        super(DurationPredictorGRU, self).__init__()
        self.sid_embedding = nn.Embedding(NUM_SINGERS, SID_EMBEDDING_DIM)
        gru_input_dim = 1 + SID_EMBEDDING_DIM
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=GRU_UNITS,
            num_layers=NUM_GRU_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT_RATE if NUM_GRU_LAYERS > 1 else 0
        )
        self.fc_out = nn.Linear(GRU_UNITS * 2, 1)

    def forward(self, duration_input, sid_input):
        sid_embedded = self.sid_embedding(sid_input)
        duration_reshaped = duration_input.unsqueeze(-1)
        x = torch.cat([duration_reshaped, sid_embedded], dim=-1)
        gru_output, _ = self.gru(x)
        predictions = self.fc_out(gru_output)
        return predictions

# --- 2. 데이터 준비 ---
print("--- 데이터셋 준비 중... ---")
# 가상의 전체 데이터셋 생성
durations = torch.rand(NUM_SAMPLES, MAX_SEQ_LENGTH)
sids = torch.randint(0, NUM_SINGERS, (NUM_SAMPLES, MAX_SEQ_LENGTH))
labels = (durations * torch.rand_like(durations) * 2).unsqueeze(-1)
full_dataset = TensorDataset(durations, sids, labels)

# 훈련(80%), 검증(10%), 테스트(10%) 데이터셋으로 분할
train_size = int(0.8 * NUM_SAMPLES)
val_size = int(0.1 * NUM_SAMPLES)
test_size = NUM_SAMPLES - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# 각 데이터셋을 위한 DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print(f"훈련 데이터 샘플 수: {len(train_dataset)}")
print(f"검증 데이터 샘플 수: {len(val_dataset)}")
print(f"테스트 데이터 샘플 수: {len(test_dataset)}")

# --- 3. 훈련 및 검증 루프 ---
print("\n--- 모델 훈련 시작... ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DurationPredictorGRU().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

# 최적 모델 저장을 위한 변수 초기화
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

for epoch in range(NUM_EPOCHS):
    # --- 훈련 단계 ---
    model.train() # 모델을 훈련 모드로 설정
    total_train_loss = 0
    for batch_idx, (duration, sid, label) in enumerate(train_loader):
        duration, sid, label = duration.to(device), sid.to(device), label.to(device)

        # 순전파
        predictions = model(duration, sid)
        loss = loss_fn(predictions, label)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_loader)

    # --- 검증 단계 ---
    model.eval() # 모델을 평가 모드로 설정
    total_val_loss = 0
    with torch.no_grad(): # 그래디언트 계산 비활성화
        for duration, sid, label in val_loader:
            duration, sid, label = duration.to(device), sid.to(device), label.to(device)
            predictions = model(duration, sid)
            loss = loss_fn(predictions, label)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    # --- 체크포인팅 및 조기 종료 로직 ---
    # 검증 손실이 개선되었는지 확인
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        # 가장 좋은 모델의 가중치를 deepcopy로 저장
        best_model_state = copy.deepcopy(model.state_dict())
        patience_counter = 0 # 인내심 카운터 초기화
        print(f"  -> 검증 성능 개선! 최적 모델 저장됨. (Val Loss: {best_val_loss:.6f})")
    else:
        patience_counter += 1
        print(f"  -> 검증 성능 개선 없음. (Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE})")

    # 조기 종료 조건 확인
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"\n조기 종료: {EARLY_STOPPING_PATIENCE} 에포크 동안 검증 성능 개선이 없어 훈련을 중단합니다.")
        break

# --- 4. 최종 모델 평가 (테스트 단계) ---
print("\n--- 최종 모델 평가 시작 (테스트 데이터셋 사용)... ---")

# 저장된 최적의 모델 가중치를 불러오기
if best_model_state:
    model.load_state_dict(best_model_state)
else:
    print("경고: 저장된 최적 모델이 없습니다. 마지막 에포크 모델로 평가합니다.")

model.eval()
total_test_loss = 0
with torch.no_grad():
    for duration, sid, label in test_loader:
        duration, sid, label = duration.to(device), sid.to(device), label.to(device)
        predictions = model(duration, sid)
        loss = loss_fn(predictions, label)
        total_test_loss += loss.item()

avg_test_loss = total_test_loss / len(test_loader)

print("-" * 50)
print(f"최종 테스트 손실 (MSE): {avg_test_loss:.6f}")
print("이것이 논문에 보고할 최종 모델의 일반화 성능입니다.")
print("-" * 50)

# (선택) 최적 모델 가중치 파일로 저장
if best_model_state:
    torch.save(best_model_state, 'best_duration_predictor.pth')
    print("최적 모델 가중치가 'best_duration_predictor.pth' 파일로 저장되었습니다.")

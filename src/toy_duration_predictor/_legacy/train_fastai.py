# --- 0. 필요 라이브러리 설치 ---
# 이 스크립트를 실행하기 전에 먼저 터미널에서 아래 명령어를 실행해주세요.
# pip install torch fastai wandb gradio

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from fastai.data.core import DataLoaders
from fastai.learner import Learner, pérdida_Calculada
from fastai.callback.wandb import WandbCallback
from fastai.callback.schedule import lr_find
import numpy as np
import pandas as pd
import gradio as gr
import os

# --- 1. 하이퍼파라미터 및 상수 정의 ---
MAX_SEQ_LENGTH = 32
NUM_SINGERS = 100
NUM_SAMPLES = 100000
BATCH_SIZE = 256

# 모델 구조 관련 파라미터 (fastai Learner에 전달)
SID_EMBEDDING_DIM = 16
GRU_UNITS = 128
NUM_GRU_LAYERS = 2

# --- 2. PyTorch 모델 아키텍처 정의 (fastai는 순수 PyTorch 모델을 그대로 사용) ---
class DurationPredictorGRU(nn.Module):
    """
    fastai의 Learner가 래핑할 순수 PyTorch 모델.
    """
    def __init__(self, num_singers, sid_embedding_dim, gru_units, num_gru_layers):
        super().__init__()
        self.sid_embedding = nn.Embedding(num_singers, sid_embedding_dim)
        gru_input_dim = 1 + sid_embedding_dim
        self.gru = nn.GRU(
            gru_input_dim, gru_units, num_gru_layers,
            batch_first=True, bidirectional=True
        )
        self.fc_out = nn.Linear(gru_units * 2, 1)

    def forward(self, x):
        # fastai는 입력을 튜플로 묶어서 전달합니다.
        duration_input, sid_input = x
        sid_embedded = self.sid_embedding(sid_input)
        duration_reshaped = duration_input.unsqueeze(-1)
        
        features = torch.cat([duration_reshaped, sid_embedded], dim=-1)
        gru_output, _ = self.gru(features)
        predictions = self.fc_out(gru_output)
        return predictions

# --- 3. 데이터 준비 ---
print("--- 데이터셋 준비 중... ---")

# 가상의 전체 데이터셋 생성 (DataFrame으로 관리하면 편리)
data = {
    'durations': [torch.rand(MAX_SEQ_LENGTH) for _ in range(NUM_SAMPLES)],
    'sids': [torch.randint(0, NUM_SINGERS, (MAX_SEQ_LENGTH,)) for _ in range(NUM_SAMPLES)],
    'labels': [d * torch.rand_like(d) * 2 for d in [d['durations'] for d in [{'durations': data} for data in [{'durations': torch.rand(MAX_SEQ_LENGTH)}] * NUM_SAMPLES]]]
}
df = pd.DataFrame(data)

# 훈련(80%), 검증(10%), 테스트(10%) 인덱스 생성
np.random.seed(42)
indices = np.random.permutation(len(df))
test_split_idx = int(len(df) * 0.1)
val_split_idx = int(len(df) * 0.2)

test_indices = indices[:test_split_idx]
val_indices = indices[test_split_idx:val_split_idx]
train_indices = indices[val_split_idx:]

# fastai의 DataLoaders 객체 생성
# 입력(x)은 튜플, 출력(y)은 단일 텐서로 구성
train_ds = TensorDataset(torch.stack(df.loc[train_indices, 'durations'].tolist()),
                         torch.stack(df.loc[train_indices, 'sids'].tolist()),
                         torch.stack(df.loc[train_indices, 'labels'].tolist()).unsqueeze(-1))

val_ds = TensorDataset(torch.stack(df.loc[val_indices, 'durations'].tolist()),
                       torch.stack(df.loc[val_indices, 'sids'].tolist()),
                       torch.stack(df.loc[val_indices, 'labels'].tolist()).unsqueeze(-1))

test_ds = TensorDataset(torch.stack(df.loc[test_indices, 'durations'].tolist()),
                        torch.stack(df.loc[test_indices, 'sids'].tolist()),
                        torch.stack(df.loc[test_indices, 'labels'].tolist()).unsqueeze(-1))

# fastai의 DataLoaders로 래핑
# 입력(x)을 튜플로 묶기 위해 x_cat=2
dls = DataLoaders.from_dsets(train_ds, val_ds, bs=BATCH_SIZE, device='cuda' if torch.cuda.is_available() else 'cpu')
test_dl = dls.test_dl(test_ds, with_labels=True)

print(f"훈련 데이터 샘플 수: {len(train_ds)}")
print(f"검증 데이터 샘플 수: {len(val_ds)}")
print(f"테스트 데이터 샘플 수: {len(test_ds)}")

# --- 4. fastai Learner 생성 및 훈련 ---

# 모델 인스턴스화
model = DurationPredictorGRU(NUM_SINGERS, SID_EMBEDDING_DIM, GRU_UNITS, NUM_GRU_LAYERS)

# Learner 생성 (모델, 데이터, 손실 함수, 콜백 등을 모두 묶음)
learn = Learner(dls, model, loss_func=nn.MSELoss(), cbs=WandbCallback(log_preds=False))

# --- 4a. 최적의 학습률 탐색 (Optuna 대신 사용) ---
print("\n--- 1. 최적의 학습률 탐색 시작 (fastai lr_find) ---")
# lr_find() 실행 후, 가장 가파른 기울기를 가진 지점의 학습률을 사용하는 것이 일반적
suggested_lr = learn.lr_find(suggest_funcs=(lr_find.valley, lr_find.slide))
print(f"fastai가 제안하는 최적 학습률: {suggested_lr.valley:.2e}")

# --- 4b. 모델 훈련 ---
print("\n--- 2. 제안된 학습률로 모델 훈련 시작 ---")
# fine_tune은 헤드는 제안된 학습률로, 몸통은 더 낮은 학습률로 훈련하는 등
# 여러 best practice가 적용된 강력한 훈련 메소드
learn.fine_tune(10, base_lr=suggested_lr.valley)

print("모델 훈련 완료!")

# --- 5. 최종 성능 평가 (테스트셋) ---
print("\n--- 3. 최종 모델 평가 시작 (테스트 데이터셋 사용) ---")
# get_preds를 사용하여 테스트셋에 대한 예측 및 손실 계산
preds, targs, test_loss = learn.get_preds(dl=test_dl, with_loss=True)
print(f"최종 테스트 손실 (MSE): {test_loss.item():.6f}")

# --- 6. Gradio 데모 실행 ---
print("\n--- 4. Gradio 데모 인터페이스 실행 ---")
learn.model.eval() # 추론을 위해 모델을 평가 모드로 전환

def predict_duration_fastai(singer_id_str, duration_sequence_str):
    try:
        # 입력 파싱 및 텐서화
        singer_id = int(singer_id_str)
        durations = [float(d.strip()) for d in duration_sequence_str.split(',')]
        
        if len(durations) > MAX_SEQ_LENGTH:
            durations = durations[:MAX_SEQ_LENGTH]
        else:
            durations += [0] * (MAX_SEQ_LENGTH - len(durations))

        duration_tensor = torch.tensor(durations, dtype=torch.float32).unsqueeze(0)
        sid_tensor = torch.full_like(duration_tensor, singer_id, dtype=torch.long)

        # fastai Learner를 사용한 예측
        # learn.predict는 단일 아이템에 대한 예측과 디코딩을 수행
        # 여기서는 모델 직접 호출이 더 간단
        with torch.no_grad():
            prediction = learn.model((duration_tensor.to(learn.dls.device), sid_tensor.to(learn.dls.device)))
        
        output_sequence = prediction.squeeze().cpu().tolist()
        return ", ".join([f"{x:.4f}" for x in output_sequence])

    except Exception as e:
        return f"오류 발생: {e}"

# Gradio 인터페이스 생성 및 실행
iface = gr.Interface(
    fn=predict_duration_fastai,
    inputs=[
        gr.Textbox(label="가수 ID (Singer ID)", value="10"),
        gr.Textbox(label="음표 길이 시퀀스 (쉼표로 구분)", 
                   value="0.1, 0.2, 0.15, 0.5, 0.4, 0.12, 0.1, 0.25")
    ],
    outputs=gr.Textbox(label="예측된 음표 길이 시퀀스"),
    title="🎵 Duration Predictor (fastai + MLOps)",
    description="fastai로 훈련된 모델입니다. 가수 ID와 정규 음표 길이 시퀀스를 입력하면, 해당 가수의 고유한 리듬 표현이 적용된 음표 길이를 예측합니다."
)
iface.launch()


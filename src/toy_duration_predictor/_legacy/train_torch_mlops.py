# --- 0. 필요 라이브러리 설치 ---
# 이 스크립트를 실행하기 전에 먼저 터미널에서 아래 명령어를 실행해주세요.
# pip install torch numpy pytorch-lightning wandb optuna gradio

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import optuna
import gradio as gr
import os

# --- 1. 가상 데이터 생성 및 PyTorch Lightning 데이터 모듈 (업데이트) ---
# 실제로는 이 부분에 MIDI 데이터를 전처리하고 로드하는 코드가 들어갑니다.

# 데이터 관련 상수
MAX_SEQ_LENGTH = 32
NUM_SINGERS = 100
NUM_SAMPLES = 100000  # 전체 샘플 수 (10만개로 증가)
BATCH_SIZE = 256      # 배치 크기

class DurationDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.full_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # 이 메소드는 단일 프로세스에서만 실행됩니다.
        # 데이터를 다운로드하거나 생성하는 로직을 여기에 넣습니다.
        pass

    def setup(self, stage=None):
        # 모든 GPU/TPU에서 실행됩니다. 데이터를 분할하고 할당합니다.
        if not self.full_dataset:
            # 가상의 전체 데이터셋 생성
            durations = torch.rand(NUM_SAMPLES, MAX_SEQ_LENGTH)
            sids = torch.randint(0, NUM_SINGERS, (NUM_SAMPLES, MAX_SEQ_LENGTH))
            labels = durations * torch.rand_like(durations) * 2
            self.full_dataset = TensorDataset(durations, sids, labels)

        # 훈련(80%), 검증(10%), 테스트(10%) 데이터셋으로 분할
        train_size = int(0.8 * len(self.full_dataset))
        val_size = int(0.1 * len(self.full_dataset))
        test_size = len(self.full_dataset) - train_size - val_size
        
        # random_split을 사용하여 데이터를 나눔 (매번 동일한 분할을 위해 시드 고정)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count()//2)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=os.cpu_count()//2)
        
    def test_dataloader(self):
        # 테스트 데이터로더 추가
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=os.cpu_count()//2)

# --- 2. PyTorch Lightning 모델 (업데이트) ---
# 테스트 스텝 추가

class DurationPredictor(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.sid_embedding = nn.Embedding(self.hparams.num_singers, self.hparams.sid_embedding_dim)
        gru_input_dim = 1 + self.hparams.sid_embedding_dim
        self.gru = nn.GRU(gru_input_dim, self.hparams.gru_units, self.hparams.num_gru_layers, 
                          batch_first=True, bidirectional=True, 
                          dropout=self.hparams.dropout_rate if self.hparams.num_gru_layers > 1 else 0)
        self.fc_out = nn.Linear(self.hparams.gru_units * 2, 1)
        self.loss_fn = nn.MSELoss()

    def forward(self, duration_input, sid_input):
        sid_embedded = self.sid_embedding(sid_input)
        duration_reshaped = duration_input.unsqueeze(-1)
        x = torch.cat([duration_reshaped, sid_embedded], dim=-1)
        gru_output, _ = self.gru(x)
        predictions = self.fc_out(gru_output)
        return predictions

    def _common_step(self, batch, batch_idx):
        durations, sids, labels = batch
        predictions = self.forward(durations, sids)
        loss = self.loss_fn(predictions.squeeze(-1), labels)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

# --- 3. Optuna를 사용한 하이퍼파라미터 최적화 ---
def objective(trial: optuna.Trial):
    hparams = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'sid_embedding_dim': trial.suggest_categorical('sid_embedding_dim', [8, 16, 32]),
        'gru_units': trial.suggest_categorical('gru_units', [64, 128]),
        'num_gru_layers': trial.suggest_int('num_gru_layers', 1, 2),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.4),
        'num_singers': NUM_SINGERS
    }
    
    wandb_logger = WandbLogger(project="duration_predictor_optuna", name=f"trial-{trial.number}", group="optuna-study")
    wandb_logger.log_hyperparams(hparams)

    model = DurationPredictor(hparams)
    datamodule = DurationDataModule(batch_size=BATCH_SIZE)

    trainer = pl.Trainer(
        logger=wandb_logger, max_epochs=5, accelerator="auto", devices=1,
        enable_checkpointing=False, callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=2)]
    )
    trainer.fit(model, datamodule)
    return trainer.callback_metrics["val_loss"].item()

# --- 4. 메인 실행 블록 (업데이트) ---
if __name__ == '__main__':
    # --- 1. 하이퍼파라미터 최적화 ---
    print("--- 1. 하이퍼파라미터 최적화 시작 (Optuna) ---")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10) # 실제로는 50~100회 이상 권장

    print("최적화 완료!")
    best_hparams = study.best_params
    print(f"최고의 val_loss: {study.best_value}\n최적 하이퍼파라미터: {best_hparams}")
    
    # --- 2. 최적 모델 훈련 및 저장 ---
    print("\n--- 2. 최적의 하이퍼파라미터로 최종 모델 학습 및 평가 시작 ---")
    final_hparams = best_hparams
    final_hparams['num_singers'] = NUM_SINGERS
    
    datamodule = DurationDataModule(batch_size=BATCH_SIZE)
    model = DurationPredictor(final_hparams)
    
    wandb_logger = WandbLogger(project="duration_predictor_final", name="final_best_model")
    wandb_logger.log_hyperparams(final_hparams)

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints', filename='best-model', save_top_k=1, monitor='val_loss', mode='min'
    )
    
    trainer = pl.Trainer(
        logger=wandb_logger, max_epochs=20, accelerator="auto", devices=1,
        callbacks=[checkpoint_callback, EarlyStopping(monitor="val_loss", mode="min", patience=4)]
    )
    trainer.fit(model, datamodule)
    print("최종 모델 학습 완료!")
    
    # --- 3. 최종 성능 평가 (테스트) ---
    print(f"저장된 최고 성능 모델 경로: {checkpoint_callback.best_model_path}")
    # trainer.test()는 최고의 체크포인트를 자동으로 불러와 평가를 진행합니다.
    test_results = trainer.test(ckpt_path='best', datamodule=datamodule)
    print("최종 테스트 결과:", test_results)
    
    # --- 4. Gradio 데모 실행 ---
    print("\n--- 4. Gradio 데모 인터페이스 실행 ---")
    
    best_model = DurationPredictor.load_from_checkpoint(checkpoint_callback.best_model_path)
    best_model.eval()

    def predict_duration(singer_id_str, duration_sequence_str):
        try:
            singer_id = int(singer_id_str)
            durations = [float(d.strip()) for d in duration_sequence_str.split(',')]
            
            if len(durations) > MAX_SEQ_LENGTH:
                durations = durations[:MAX_SEQ_LENGTH]
            else:
                durations += [0] * (MAX_SEQ_LENGTH - len(durations))

            duration_tensor = torch.tensor(durations, dtype=torch.float32).unsqueeze(0)
            sid_tensor = torch.full_like(duration_tensor, singer_id, dtype=torch.long)

            with torch.no_grad():
                prediction = best_model(duration_tensor, sid_tensor)
            
            output_sequence = prediction.squeeze().tolist()
            return ", ".join([f"{x:.4f}" for x in output_sequence])

        except Exception as e:
            return f"오류 발생: {e}"

    iface = gr.Interface(
        fn=predict_duration,
        inputs=[
            gr.Textbox(label="가수 ID (Singer ID)", value="10"),
            gr.Textbox(label="음표 길이 시퀀스 (쉼표로 구분)", 
                       value="0.1, 0.2, 0.15, 0.5, 0.4, 0.12, 0.1, 0.25")
        ],
        outputs=gr.Textbox(label="예측된 음표 길이 시퀀스"),
        title="🎵 Duration Predictor (리듬 표현 예측기)",
        description="가수 ID와 정규 음표 길이 시퀀스를 입력하면, 해당 가수의 고유한 리듬 표현이 적용된 음표 길이를 예측합니다."
    )
    
    iface.launch()

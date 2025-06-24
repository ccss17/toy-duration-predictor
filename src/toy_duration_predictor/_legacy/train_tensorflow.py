# --- 0. 필요 라이브러리 설치 ---
# 이 스크립트를 실행하기 전에 먼저 터미널에서 아래 명령어를 실행해주세요.
# pip install tensorflow numpy wandb keras-tuner gradio

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
import keras_tuner as kt
from wandb.keras import WandbCallback
import gradio as gr
import os

# --- 1. 하이퍼파라미터 및 상수 정의 ---
MAX_SEQ_LENGTH = 32
NUM_SINGERS = 100
NUM_SAMPLES = 100000
BATCH_SIZE = 256
BUFFER_SIZE = 10000 # tf.data.Dataset 셔플을 위한 버퍼 크기

# --- 2. 데이터 준비 (tf.data.Dataset 사용) ---
print("--- 데이터셋 준비 중... ---")

def generate_dummy_data():
    """가상의 데이터셋을 생성하는 제너레이터 함수"""
    for _ in range(NUM_SAMPLES):
        duration = np.random.rand(MAX_SEQ_LENGTH).astype(np.float32)
        sid = np.random.randint(0, NUM_SINGERS, (MAX_SEQ_LENGTH,)).astype(np.int32)
        label = (duration * np.random.rand(MAX_SEQ_LENGTH) * 2).astype(np.float32)
        # Keras 모델은 입력과 출력을 딕셔너리 형태로 받는 것이 편리합니다.
        yield {'duration_input': duration, 'sid_input': sid}, label

# tf.data.Dataset 객체 생성
full_dataset = tf.data.Dataset.from_generator(
    generate_dummy_data,
    output_signature=(
        {'duration_input': tf.TensorSpec(shape=(MAX_SEQ_LENGTH,), dtype=tf.float32),
         'sid_input': tf.TensorSpec(shape=(MAX_SEQ_LENGTH,), dtype=tf.int32)},
        tf.TensorSpec(shape=(MAX_SEQ_LENGTH,), dtype=tf.float32)
    )
)

# 훈련(80%), 검증(10%), 테스트(10%) 데이터셋으로 분할
full_dataset = full_dataset.shuffle(BUFFER_SIZE, seed=42) # 분할 전 전체 셔플
train_size = int(0.8 * NUM_SAMPLES)
val_size = int(0.1 * NUM_SAMPLES)

train_dataset = full_dataset.take(train_size)
val_and_test_dataset = full_dataset.skip(train_size)
val_dataset = val_and_test_dataset.take(val_size)
test_dataset = val_and_test_dataset.skip(val_size)

# 데이터로더 생성 (배치, 프리페치 등 최적화)
train_loader = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_loader = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_loader = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"훈련 데이터 샘플 수: {train_size}")
print(f"검증 데이터 샘플 수: {val_size}")
print(f"테스트 데이터 샘플 수: {NUM_SAMPLES - train_size - val_size}")

# --- 3. KerasTuner를 사용한 하이퍼파라미터 최적화 ---

def build_model(hp: kt.HyperParameters):
    """KerasTuner가 하이퍼파라미터를 탐색하기 위한 모델 빌드 함수"""
    
    # 입력층 정의
    duration_input = layers.Input(shape=(MAX_SEQ_LENGTH,), name='duration_input')
    sid_input = layers.Input(shape=(MAX_SEQ_LENGTH,), name='sid_input')

    # 하이퍼파라미터 탐색 공간 정의
    sid_embedding_dim = hp.Choice('sid_embedding_dim', values=[8, 16, 32])
    gru_units = hp.Choice('gru_units', values=[64, 128])
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    
    # 모델 레이어
    sid_embedding = layers.Embedding(input_dim=NUM_SINGERS, output_dim=sid_embedding_dim)(sid_input)
    duration_reshaped = layers.Reshape((MAX_SEQ_LENGTH, 1))(duration_input)
    
    x = layers.Concatenate()([duration_reshaped, sid_embedding])
    
    # Keras의 GRU는 기본적으로 dropout 인자를 가짐
    x = layers.Bidirectional(layers.GRU(gru_units, return_sequences=True))(x)
    x = layers.Bidirectional(layers.GRU(gru_units, return_sequences=True))(x)
    
    outputs = layers.TimeDistributed(layers.Dense(1, activation='linear'))(x)

    model = Model(inputs=[duration_input, sid_input], outputs=outputs)
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='mean_squared_error')
    
    return model

print("\n--- 1. 하이퍼파라미터 최적화 시작 (KerasTuner) ---")
tuner = kt.Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=10,
    factor=3,
    directory='keras_tuner_dir',
    project_name='duration_predictor'
)

# KerasTuner 실행
tuner.search(train_loader, epochs=10, validation_data=val_loader, callbacks=[keras.callbacks.EarlyStopping(patience=3)])

# 최적 하이퍼파라미터 추출
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("최적화 완료!")
print(f"최적의 학습률: {best_hps.get('learning_rate')}")
print(f"최적의 임베딩 차원: {best_hps.get('sid_embedding_dim')}")
print(f"최적의 GRU 유닛 수: {best_hps.get('gru_units')}")

# --- 4. 최종 모델 훈련 및 평가 ---
print("\n--- 2. 최적의 하이퍼파라미터로 최종 모델 학습 및 평가 시작 ---")

# WandB 초기화
import wandb
wandb.init(project="duration_predictor_tf_keras", config=best_hps.values)

# 최적의 하이퍼파라미터로 최종 모델 빌드
final_model = tuner.hypermodel.build(best_hps)

# 체크포인트 및 조기 종료 콜백 설정
checkpoint_cb = keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

# 모델 훈련
final_model.fit(
    train_loader,
    epochs=50,
    validation_data=val_loader,
    callbacks=[WandbCallback(), checkpoint_cb, early_stopping_cb]
)
print("최종 모델 학습 완료!")

# --- 5. 최종 성능 평가 (테스트셋) ---
print("\n--- 3. 최종 모델 평가 시작 (테스트 데이터셋 사용) ---")
best_model = keras.models.load_model("best_model.keras")
test_loss = best_model.evaluate(test_loader)
print("-" * 50)
print(f"최종 테스트 손실 (MSE): {test_loss:.6f}")
wandb.log({"test_loss": test_loss})
wandb.finish()

# --- 6. Gradio 데모 실행 ---
print("\n--- 4. Gradio 데모 인터페이스 실행 ---")

def predict_duration_keras(singer_id_str, duration_sequence_str):
    try:
        singer_id = int(singer_id_str)
        durations = [float(d.strip()) for d in duration_sequence_str.split(',')]
        
        if len(durations) > MAX_SEQ_LENGTH:
            durations = durations[:MAX_SEQ_LENGTH]
        else:
            durations += [0] * (MAX_SEQ_LENGTH - len(durations))

        # Keras 모델 입력 형태로 변환 (Numpy 배열)
        duration_np = np.array(durations, dtype=np.float32).reshape(1, -1)
        sid_np = np.full_like(duration_np, singer_id, dtype=np.int32)

        # 예측 실행
        prediction = best_model.predict({'duration_input': duration_np, 'sid_input': sid_np})
        
        output_sequence = prediction.flatten().tolist()
        return ", ".join([f"{x:.4f}" for x in output_sequence])
    except Exception as e:
        return f"오류 발생: {e}"

iface = gr.Interface(
    fn=predict_duration_keras,
    inputs=[
        gr.Textbox(label="가수 ID (Singer ID)", value="10"),
        gr.Textbox(label="음표 길이 시퀀스 (쉼표로 구분)", 
                   value="0.1, 0.2, 0.15, 0.5, 0.4, 0.12, 0.1, 0.25")
    ],
    outputs=gr.Textbox(label="예측된 음표 길이 시퀀스"),
    title="🎵 Duration Predictor (Keras + MLOps)",
    description="Keras로 훈련된 모델입니다. 가수 ID와 정규 음표 길이 시퀀스를 입력하면, 해당 가수의 고유한 리듬 표현이 적용된 음표 길이를 예측합니다."
)

iface.launch()

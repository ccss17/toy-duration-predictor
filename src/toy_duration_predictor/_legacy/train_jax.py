# --- 0. 필요 라이브러리 설치 ---
# 이 스크립트를 실행하기 전에 먼저 터미널에서 아래 명령어를 실행해주세요.
# CPU 버전: pip install jax flax optax elegy
# GPU 버전: pip install jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
#           pip install flax optax elegy

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import elegy
import numpy as np
from sklearn.model_selection import train_test_split

# --- 1. 하이퍼파라미터 및 상수 정의 ---
MAX_SEQ_LENGTH = 32
NUM_SINGERS = 100
NUM_SAMPLES = 100000
BATCH_SIZE = 256

# 모델 구조 관련 파라미터
SID_EMBEDDING_DIM = 16
GRU_UNITS = 128
NUM_GRU_LAYERS = 2  # Flax의 GRUCell은 num_layers를 직접 지원하지 않으므로, 루프로 구현해야 합니다.
                  # 이 예제에서는 간결성을 위해 1개 층으로 구현합니다.
LEARNING_RATE = 0.001
NUM_EPOCHS = 20

# --- 2. Flax를 사용한 모델 아키텍처 정의 ---
class DurationPredictorGRU(nn.Module):
    """
    Flax를 사용하여 정의한 Duration Predictor 모델.
    JAX의 순수 함수 철학을 따릅니다.
    """
    num_singers: int
    sid_embedding_dim: int
    gru_units: int
    
    @nn.compact
    def __call__(self, x):
        # Elegy는 입력을 튜플/리스트 대신 딕셔너리로 받는 것을 선호합니다.
        duration_input = x['duration_input']
        sid_input = x['sid_input']

        # 1. SID 임베딩
        sid_embedded = nn.Embed(
            num_embeddings=self.num_singers,
            features=self.sid_embedding_dim
        )(sid_input)

        # 2. 음표 길이 차원 확장
        duration_reshaped = jnp.expand_dims(duration_input, axis=-1)
        
        # 3. 피처 연결
        features = jnp.concatenate([duration_reshaped, sid_embedded], axis=-1)

        # 4. 양방향 GRU
        # Flax의 Bidirectional 래퍼는 RNNCell을 감싸서 양방향으로 만듭니다.
        gru_cell = nn.GRUCell(features=self.gru_units)
        gru_output = nn.Bidirectional(gru_cell)(features)
        
        # 5. 출력층
        # Flax의 Dense는 시퀀스 입력에 대해 자동으로 Time-Distributed처럼 작동합니다.
        predictions = nn.Dense(features=1)(gru_output)
        
        return predictions

# --- 3. 데이터 준비 ---
print("--- 데이터셋 준비 중... ---")

# 가상의 Numpy 데이터셋 생성
durations = np.random.rand(NUM_SAMPLES, MAX_SEQ_LENGTH).astype(np.float32)
sids = np.random.randint(0, NUM_SINGERS, (NUM_SAMPLES, MAX_SEQ_LENGTH)).astype(np.int32)
labels = (durations * np.random.rand(NUM_SAMPLES, MAX_SEQ_LENGTH) * 2).astype(np.float32)

# 훈련(80%), 검증/테스트(20%)로 먼저 분할
dur_train, dur_rem, sids_train, sids_rem, y_train, y_rem = train_test_split(
    durations, sids, labels, test_size=0.2, random_state=42)

# 검증(10%), 테스트(10%)로 분할
dur_val, dur_test, sids_val, sids_test, y_val, y_test = train_test_split(
    dur_rem, sids_rem, y_rem, test_size=0.5, random_state=42)

# Elegy가 사용할 수 있도록 입력 데이터를 딕셔너리 형태로 묶습니다.
X_train = {'duration_input': dur_train, 'sid_input': sids_train}
X_val = {'duration_input': dur_val, 'sid_input': sids_val}
X_test = {'duration_input': dur_test, 'sid_input': sids_test}

print(f"훈련 데이터 샘플 수: {len(y_train)}")
print(f"검증 데이터 샘플 수: {len(y_val)}")
print(f"테스트 데이터 샘플 수: {len(y_test)}")

# --- 4. Elegy를 사용한 모델 훈련 및 평가 ---

# Elegy 모델 생성
# Keras와 매우 유사하게, 모듈, 손실함수, 옵티마이저, 메트릭을 정의합니다.
model = elegy.Model(
    module=DurationPredictorGRU(
        num_singers=NUM_SINGERS,
        sid_embedding_dim=SID_EMBEDDING_DIM,
        gru_units=GRU_UNITS
    ),
    loss=elegy.losses.MeanSquaredError(),
    optimizer=optax.adam(learning_rate=LEARNING_RATE),
    metrics=[elegy.metrics.MeanAbsoluteError()]
)

print("\n--- 모델 훈련 시작 (Elegy)... ---")
# Keras와 거의 동일한 .fit() API를 사용
# (참고: WandB, Gradio 등은 Elegy의 콜백 시스템을 통해 연동 가능합니다)
history = model.fit(
    x=X_train,
    y=y_train,
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=[elegy.callbacks.EarlyStopping(monitor='val_loss', patience=5)],
    shuffle=True
)

print("모델 훈련 완료!")

# --- 5. 최종 성능 평가 (테스트셋) ---
print("\n--- 최종 모델 평가 시작 (테스트 데이터셋 사용)... ---")

# .evaluate() API를 사용하여 최종 성능 측정
test_metrics = model.evaluate(X_test, y_test)
print("-" * 50)
print(f"최종 테스트 결과: {test_metrics}")
print("-" * 50)

# --- 6. Gradio 데모를 위한 예측 함수 (예시) ---
# Elegy 모델은 내부적으로 JAX의 JIT 컴파일을 사용하여 예측 속도가 매우 빠릅니다.
@jax.jit
def predict_fn(params, x):
    return model.module.apply({'params': params}, x)

def gradio_predict(singer_id_str, duration_sequence_str):
    try:
        singer_id = int(singer_id_str)
        durations = [float(d.strip()) for d in duration_sequence_str.split(',')]
        
        if len(durations) > MAX_SEQ_LENGTH:
            durations = durations[:MAX_SEQ_LENGTH]
        else:
            durations += [0] * (MAX_SEQ_LENGTH - len(durations))

        duration_np = np.array(durations, dtype=np.float32).reshape(1, -1)
        sid_np = np.full_like(duration_np, singer_id, dtype=np.int32)
        
        input_dict = {'duration_input': duration_np, 'sid_input': sid_np}
        
        # JIT 컴파일된 함수로 예측 실행
        prediction = predict_fn(model.states.params, input_dict)
        
        output_sequence = np.asarray(prediction).flatten().tolist()
        return ", ".join([f"{x:.4f}" for x in output_sequence])
    except Exception as e:
        return f"오류 발생: {e}"

# (Gradio 실행 부분은 주석 처리. 필요시 주석 해제하여 사용)
# print("\n--- Gradio 데모 인터페이스 실행 ---")
# iface = gr.Interface(
#     fn=gradio_predict,
#     inputs=[
#         gr.Textbox(label="가수 ID (Singer ID)", value="10"),
#         gr.Textbox(label="음표 길이 시퀀스 (쉼표로 구분)", value="0.1, 0.2, 0.15, 0.5")
#     ],
#     outputs=gr.Textbox(label="예측된 음표 길이 시퀀스"),
#     title="🎵 Duration Predictor (JAX/Flax/Elegy)",
#     description="JAX 생태계로 훈련된 모델입니다. 가수 ID와 정규 음표 길이 시퀀스를 입력하면, 해당 가수의 고유한 리듬 표현이 적용된 음표 길이를 예측합니다."
# )
# iface.launch()

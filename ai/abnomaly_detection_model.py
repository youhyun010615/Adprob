import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# === 데이터 로딩 ===
df = pd.read_csv("ai/output.csv")

# === 비광고 샘플만 ===
df_normal = df[df['is_ad'] == 0]
X = df_normal.drop(columns=["text", "is_ad", "image_count"]).fillna(0)

# === 정규화 ===
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# === LSTM 입력형태로 변환 ===
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))  # (samples, timesteps=1, features)

# === LSTM AutoEncoder 모델 정의 ===
input_dim = X_scaled.shape[2]
input_layer = Input(shape=(1, input_dim))
encoded = LSTM(64, activation='relu', return_sequences=False)(input_layer)
bottleneck = RepeatVector(1)(encoded)
decoded = LSTM(64, activation='relu', return_sequences=True)(bottleneck)
output_layer = TimeDistributed(Dense(input_dim))(decoded)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='mse')

# === 모델 학습 ===
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(
    X_scaled, X_scaled,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)

# === 모델 및 스케일러 저장 ===
model.save("ai/abnomaly_detection_model/anomaly_lstm_autoencoder.keras")
joblib.dump(scaler, "ai/abnomaly_detection_model/scaler_lstm.pkl")

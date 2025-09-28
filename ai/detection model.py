import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
    precision_recall_curve
)

# --- CSV 로딩 ---
df = pd.read_csv('ai/output.csv')

# --- 타겟 및 피처 설정 ---
target_col = 'is_ad'
feature_cols = [col for col in df.columns if col not in [target_col, 'text', 'image_count']]
df[feature_cols] = df[feature_cols].fillna(0)

# --- 학습/테스트 분리 ---
X_train, X_test, y_train, y_test = train_test_split(
    df[feature_cols], df[target_col], test_size=0.2, random_state=42, stratify=df[target_col]
)

# --- CatBoost 모델 학습 ---
model = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    loss_function='Logloss',
    eval_metric='F1',
    verbose=50
)
model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

# --- 확률 예측 ---
y_prob = model.predict_proba(X_test)[:, 1]

# --- 임계값 기준 평가 (기본 0.5) ---
threshold = 0.5
y_pred = (y_prob >= threshold).astype(int)

print(f"\n🔍 [Threshold: {threshold}] Classification Report")
print(classification_report(y_test, y_pred, digits=4))

precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy : {acc:.4f}")
print(f"🎯 Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# --- 모델 저장 ---
model.save_model('ai/detection_model/ad_detection_model.cbm')
print("💾 모델 저장 완료")

# --- Feature Importance 시각화 ---
importances = model.get_feature_importance()
sorted_idx = np.argsort(importances)[::-1]
sorted_features = [feature_cols[i] for i in sorted_idx]

plt.figure(figsize=(10, 6))
plt.barh(sorted_features[:15][::-1], importances[sorted_idx][:15][::-1])
plt.xlabel("Importance")
plt.title("Top 15 Feature Importances")
plt.tight_layout()
plt.show()

# --- Precision-Recall Curve + 최적 threshold 탐색 ---
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"\n🔥 Best F1-score = {best_f1:.4f} at threshold = {best_threshold:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, label='PR Curve')
plt.scatter(recalls[best_idx], precisions[best_idx], color='red', label=f'Best F1 = {best_f1:.4f}')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

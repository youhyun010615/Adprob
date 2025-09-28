from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from catboost import CatBoostClassifier
from ai.feature import analyze_reviews
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import pytesseract
import base64
import joblib
from tensorflow.keras.models import load_model
from fetcher import fetch_main_container_images_only
from openai import OpenAI
import requests

app = Flask(__name__)
CORS(app)

# --- 모델 로드 ---
catboost_model = CatBoostClassifier()
catboost_model.load_model("ai/detection_model/ad_detection_model.cbm")

anomaly_model = load_model("ai/abnomaly_detection_model/anomaly_lstm_autoencoder.keras")
scaler = joblib.load("ai/abnomaly_detection_model/scaler_lstm.pkl")

def merge_images_vertically(img_urls):
    images = []
    sizes = []

    for url in img_urls:
        try:
            res = requests.get(url, timeout=5)
            img = Image.open(BytesIO(res.content))

            if img.mode != "RGBA":
                img = img.convert("RGBA")  # 알파 보존용 변환

            w, h = img.size
            sizes.append((w, h))
            images.append(img)
        except Exception as e:
            print(f"[이미지 로딩 실패] {url} / {e}")
            continue

    if not images:
        return None, []

    # 평균 (w + h)
    wh_sums = [w + h for w, h in sizes]
    avg_wh = sum(wh_sums) / len(wh_sums)
    print(f"📏 평균 (W+H): {avg_wh:.1f}")

    resized_images = []
    for img, (w, h) in zip(images, sizes):
        current_wh = w + h
        ratio = avg_wh / current_wh
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        resized = img.resize((new_w, new_h))
        resized_images.append(resized)

    max_width = max(img.width for img in resized_images)
    total_height = sum(img.height for img in resized_images)

    # RGBA 모드로 투명 배경 유지
    merged = Image.new("RGBA", (max_width, total_height), (0, 0, 0, 0))  # 완전 투명

    y_offset = 0
    for img in resized_images:
        x_offset = int((max_width - img.width) / 2)
        merged.paste(img, (x_offset, y_offset), mask=img)  # 알파 적용
        y_offset += img.height

    # 인코딩
    buffer = BytesIO()
    merged.save(buffer, format="PNG")
    base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return base64_img, resized_images

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    url = data.get("url")

    if not url:
        return jsonify({"error": "url 누락됨"}), 400

    # ✅ fetcher로 본문 + 이미지 전부 서버에서 직접 가져오기
    fetched = fetch_main_container_images_only(url)
    text = fetched["text"]
    img_urls = fetched["images"]

    if not text or text.startswith("❌") or len(text.strip()) < 10:
        return jsonify({"error": "본문 없음"})

    # --- 피처 생성 ---
    df = pd.DataFrame({
        "text": [text],
        "image_count": [len(img_urls)]
    })
    df = analyze_reviews(df)
    feature_cols = [col for col in df.columns if col not in ['text', 'is_ad', 'image_count']]
    df[feature_cols] = df[feature_cols].fillna(0)

    client = OpenAI(api_key="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")

    # 이미지 병합 + GPT OCR 요청
    base64_img, raw_images = merge_images_vertically(img_urls)

    ocr_texts, ocr_images = [], []
    if base64_img:
        image_data_url = f"data:image/png;base64,{base64_img}"
        question = "이 이미지 내용을 텍스트로 추출해줘. 다른 말은 절대 하지 말고 안에 텍스트만 말해. 텍스트가 없으면 '텍스트를 찾을 수 없습니다' 이렇게만 말해."

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {"type": "image_url", "image_url": {"url": image_data_url}}
                        ]
                    }
                ],
                max_tokens=600
            )
            gpt_ocr = response.choices[0].message.content.strip()
            ocr_texts.append(gpt_ocr)
            ocr_images.append(base64_img)  # 병합 이미지 1장만 있음
        except Exception as e:
            ocr_texts.append(f"[GPT OCR 실패] {str(e)}")
            ocr_images.append(None)
    else:
        ocr_texts.append("[이미지 병합 실패]")
        ocr_images.append(None)

    # --- 광고 키워드 기반 판단 ---
    ad_keywords = ["협찬", "제공", "광고", "지원받아", "제공받아", "협찬받아", "후원", "홍보", "원고료", "소정의"]
    ocr_combined = " ".join(ocr_texts)
    if any(kw in ocr_combined for kw in ad_keywords):
        return jsonify({
            "label": 1,
            "prob": 1.0,
            "anomaly_detected": None,
            "anomaly_score": None,
            "ocr_texts": ocr_texts,
            "ocr_images": ocr_images,
            "imgUrls": img_urls
        })

    # --- CatBoost 예측 + 이상 탐지 ---
    prob = float(catboost_model.predict_proba(df[feature_cols])[0][1])
    anomaly_score = None
    anomaly_detected = False

    if prob <= 0.3:
        label = 0
    elif prob >= 0.7:
        label = 1
    else:
        X_scaled = scaler.transform(df[feature_cols])
        X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        recon = anomaly_model.predict(X_scaled)
        mse = float(np.mean(np.square(X_scaled - recon), axis=(1, 2))[0])
        anomaly_score = mse
        anomaly_detected = mse > 0.05
        label = 1 if anomaly_detected else 0

    return jsonify({
        "label": label,
        "prob": prob,
        "anomaly_detected": anomaly_detected,
        "anomaly_score": anomaly_score,
        "ocr_texts": ocr_texts,
        "ocr_images": ocr_images,
        "imgUrls": img_urls
    })

if __name__ == "__main__":
    app.run(port=5000, debug=True)

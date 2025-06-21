import os
import sys
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

# 프로젝트 폴더를 PYTHONPATH에 추가
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from feature_extract import parse_url_features

# Flask 앱 초기화
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
app.config["TEMPLATES_AUTO_RELOAD"] = True

# 모델 로드
# Windows 절대경로 예시
MODEL_PATH = r"C:\Users\soyun\Desktop\flask\random_forest_model.pkl"
model = joblib.load(MODEL_PATH)
print(f"[INFO] 모델 로드 성공: {MODEL_PATH} (type={type(model)})")


# 루트 페이지
@app.route("/")
def home():
    return render_template("index.html")

# 예측 API
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}
    url = (data.get("url") or "").strip()
    feats = parse_url_features(url)
    df = pd.DataFrame([feats])
    prob = model.predict_proba(df)[0][1]
    return jsonify({"bad_prob": round(prob, 3)})

# 서버 실행
if __name__ == "__main__":
    app.run(debug=True, port=8000)

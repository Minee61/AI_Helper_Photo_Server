# 라이브러리 임포트
import io, json, re
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import onnxruntime as ort

# 앱 생성
app = FastAPI(title="전자기기 분류 서버")

# CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 경로
MODEL_PATH = Path("models/squeezenet1.1-7.onnx")
IDX_PATH = Path("labels/imagenet_class_index.json")
MAP_PATH = Path("mapping/electronics_map.json")

# 모델 로드
sess = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
inp_name = sess.get_inputs()[0].name
out_name = sess.get_outputs()[0].name

# 이미지넷 라벨 로드
with IDX_PATH.open("r", encoding="utf-8") as f:
    idx_json = json.load(f)
IMAGENET_LABELS = [idx_json[str(i)][1] for i in range(len(idx_json))]

# 전자기기 매핑 로드
with MAP_PATH.open("r", encoding="utf-8") as f:
    ELECTRONICS_MAP = json.load(f)
COMPILED_RULES = {
    cat: [re.compile(rf"\b{re.escape(k.lower())}\b") for k in keys]
    for cat, keys in ELECTRONICS_MAP.items()
}

# 이미지 전처리
def preprocess(img_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, 0)

# 소프트맥스
def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    y = np.exp(x)
    return y / y.sum()

# 라벨 매핑
def label_to_electronics(label: str) -> str | None:
    L = label.lower()
    for category, rules in COMPILED_RULES.items():
        for pat in rules:
            if pat.search(L):
                return category
    return None

# 헬스체크
@app.get("/health")
def health():
    return {"status": "ok"}

# 분류 API
@app.post("/classify/")
async def classify(file: UploadFile = File(...), topk: int = 5, conf: float = 0.18):
    img_bytes = await file.read()
    x = preprocess(img_bytes)
    logits = sess.run([out_name], {inp_name: x})[0].ravel()
    probs = softmax(logits)
    top_idx = probs.argsort()[-topk:][::-1]

    candidates = [{"label": IMAGENET_LABELS[i], "prob": float(probs[i])} for i in top_idx]

    mapped = None
    top1_prob = 0.0
    for i in top_idx:
        mapped = label_to_electronics(IMAGENET_LABELS[i])
        if mapped:
            top1_prob = float(probs[i])
            break

    if not mapped or top1_prob < conf:
        return {"device": "unknown", "candidates": candidates}

    return {"device": mapped, "confidence": top1_prob, "candidates": candidates}

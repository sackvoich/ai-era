import os, time
from typing import List
import pandas as pd
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="RU Moderation (local)", version="2.0")

# После создания app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Для разработки, потом замените на конкретный origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_ID = "./models_5"
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
if not (0.0 <= THRESHOLD <= 1.0):
    raise RuntimeError("THRESHOLD must be in [0,1]")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class PredictIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)

# Загружаем модель/токенайзер один раз при старте
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.to(DEVICE)
    model.eval()
except Exception as e:
    raise RuntimeError(f"Не удалось загрузить модель {MODEL_ID}: {e}")

def _probas_local(texts: List[str], batch_size: int = 16) -> List[float]:
    probs: List[float] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=256,  # хватит для отзывов, если нет — увеличь
                return_tensors="pt"
            )
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            logits = model(**enc).logits  # [B, num_labels]
            # softmax по классам
            p = F.softmax(logits, dim=1)
            if p.size(1) == 2:
                # считаем, что класс 1 = "toxic/profanity"
                probs.extend(p[:, 1].detach().cpu().tolist())
            else:
                # если классов больше, берём максимум как «плохой» суррогат
                probs.extend(p.max(dim=1).values.detach().cpu().tolist())
    return probs

@app.get("/health")
def health():
    return {"status": "ok", "model_id": MODEL_ID, "threshold": THRESHOLD, "device": DEVICE}

@app.get("/ready")
def ready():
    # Быстрый прогоночный вызов
    try:
        _ = _probas_local(["тест локального инференса"], batch_size=1)
        return {"ready": True}
    except Exception as e:
        raise HTTPException(503, f"Model not ready: {e}")

@app.post("/predict")
def predict(payload: PredictIn):
    prob = _probas_local([payload.text])[0]
    label = int(prob >= THRESHOLD)
    return {"label": label, "score": prob, "model_version": MODEL_ID}

@app.post("/predict/csv")
def predict_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Ожидается CSV с колонками: id,text")
    try:
        df = pd.read_csv(file.file, encoding="utf-8")
    except Exception as e:
        raise HTTPException(400, f"Не удалось прочитать CSV: {e!s}")
    if not {"id", "text"}.issubset(df.columns):
        raise HTTPException(400, "CSV должен содержать колонки: id,text")

    texts = df["text"].astype(str).tolist()
    if len(texts) == 0:
        return Response("id,label\n", media_type="text/csv")

    probs = _probas_local(texts)
    labels = [int(p >= THRESHOLD) for p in probs]
    out = pd.DataFrame({"id": df["id"], "label": labels})
    return Response(out.to_csv(index=False, lineterminator="\n"), media_type="text/csv")
    #out.to_csv("predicted.csv", index=False, lineterminator="\n")  # <-- сохранение на диск
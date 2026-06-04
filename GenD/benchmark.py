import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)

from src.hf.modeling_gend import GenD

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_list(txt_path):
    paths, labels = [], []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p, y = line.split()
            paths.append(p)
            labels.append(int(y))
    return paths, np.array(labels)


@torch.no_grad()
def infer_images(txt_file, model_name):
    # load model
    model = GenD.from_pretrained(model_name)
    model.eval().to(DEVICE)
    preproc = model.feature_extractor.preprocess

    paths, y_true = load_list(txt_file)

    y_scores = []

    for p in tqdm(paths, desc="Inference"):
        img = Image.open(p).convert("RGB")
        x = preproc(img).unsqueeze(0).to(DEVICE)

        logits = model(x)
        probs = logits.softmax(dim=-1)[0]

        p_fake = float(probs[1])
        y_scores.append(p_fake)

    return np.array(y_scores), y_true


def evaluate_thresholds(y_scores, y_true, thresholds):
    rows = []

    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)

        rows.append({
            "threshold": t,
            "accuracy": acc,
            "f1": f1,
            "precision": prec,
            "recall": rec,
        })

    return rows


if __name__ == "__main__":
    TXT_FILE = "cdf.txt"
    MODEL_NAME = "yermandy/GenD_PE_L"

    y_scores, y_true = infer_images(TXT_FILE, MODEL_NAME)

    # ROC AUC (НЕ зависит от порога)
    roc_auc = roc_auc_score(y_true, y_scores)
    print(f"ROC AUC: {roc_auc:.4f}")

    # thresholds
    thresholds = np.linspace(0.0, 1.0, 101)

    rows = evaluate_thresholds(y_scores, y_true, thresholds)

    # best F1
    best = max(rows, key=lambda x: x["f1"])
    print("Best F1:")
    print(best)
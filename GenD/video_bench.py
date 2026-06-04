import cv2
import torch
import numpy as np
from PIL import Image
import imageio.v3 as iio

from src.hf.modeling_gend import GenD
from src.retinaface import prepare_model
from detector import align_face

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

@torch.no_grad()
def infer_video(
    model,
    video_path: str,
    stride: int = 1,
    face_thresh: float = 0.5,
    scale: float = 1.3,
    target_size: int | None = None,
    max_faces: int | None = 1,
):
    # 1. Load model
    preproc = model.feature_extractor.preprocess
    # 2. Load face detector
    face_detector = prepare_model(face_thresh)

    p_fake_values = []
    total_faces = 0

    for idx, frame_rgb in enumerate(iio.imiter(video_path, plugin="pyav")):
        if idx % stride != 0:
            continue

        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # 3. Face detection
        xyxy, landmarks = face_detector.detect(frame)
        if xyxy is None:
            continue

        # sort faces by area
        areas = [
            (i, (b[2] - b[0]) * (b[3] - b[1]))
            for i, b in enumerate(xyxy)
        ]
        areas.sort(key=lambda x: x[1], reverse=True)
        indices = [i for i, _ in areas[:max_faces]]

        for i in indices:
            try:
                aligned, _ = align_face(
                    frame,
                    landmarks[i],
                    scale=scale,
                    target_size=(target_size, target_size) if target_size else None,
                )
            except Exception:
                continue

            img = Image.fromarray(
                cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            )

            tensor = preproc(img).unsqueeze(0).to(DEVICE)

            logits = model(tensor)
            probs = logits.softmax(dim=-1)[0].cpu().numpy()

            p_fake = float(probs[1])
            p_fake_values.append(p_fake)
            total_faces += 1

    return {
        "num_frames": len(p_fake_values),
        "num_faces": total_faces,
        "avg_p_fake": float(np.mean(p_fake_values)) if p_fake_values else 0.0,
        "median_p_fake": float(np.median(p_fake_values)) if p_fake_values else 0.0,
    }

import csv
import os
def read_benchmark_csv(csv_path, has_header=True):
    samples = []

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        if has_header:
            next(reader)

        for row in reader:
            video_path = os.path.join("/mnt/tank/scratch/dstoronkin/LAV-DF/LAV-DF", row[0])
            label = int(row[1])
            samples.append((video_path, label))

    return samples


from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
def compute_metrics_at_thresholds(y_true, y_score, thresholds):
    results = []

    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)

        results.append({
            "threshold": thr,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        })

    return results

def compute_roc_auc(y_true, y_score):
    return roc_auc_score(y_true, y_score)


def eval_benchmark(
    model,
    samples,
    stride=1,
    max_faces=1,
):
    y_true = []
    y_score = []

    for video_path, gt in tqdm(samples):
        stats = infer_video(
            model,
            video_path,
            stride=stride,
            max_faces=max_faces,
        )

        y_true.append(gt)
        y_score.append(stats["avg_p_fake"])

    return np.array(y_true), np.array(y_score)

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = GenD.from_pretrained("yermandy/GenD_CLIP_L_14", local_files_only=True)
    model.eval().to(DEVICE)

    benchmark_csv = "/nfs/home/dstoronkin/LVLM-DFD/test_real_200_fake_300.csv"
    samples = read_benchmark_csv(benchmark_csv)

    y_true, y_score = eval_benchmark(
        model,
        samples,
        stride=1,
        max_faces=1,
    )

    roc_auc = compute_roc_auc(y_true, y_score)
    print(f"ROC-AUC: {roc_auc:.4f}")

    thresholds = np.linspace(0.1, 0.9, 9)
    metrics = compute_metrics_at_thresholds(y_true, y_score, thresholds)

    for m in metrics:
        print(
            f"thr={m['threshold']:.2f} | "
            f"acc={m['accuracy']:.3f} | "
            f"prec={m['precision']:.3f} | "
            f"rec={m['recall']:.3f} | "
            f"f1={m['f1']:.3f}"
        )


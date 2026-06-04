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


if __name__ == "__main__":
    import os
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        roc_auc_score,
        classification_report,
        confusion_matrix,
    )

    folder = Path("/mnt/tank/scratch/dstoronkin/hwei_part1")
    meta_csv = folder / "d83a0ce6-dc87-46a6-9679-98a71cf91886.csv"

    threshold = 0.5

    model = GenD.from_pretrained("yermandy/GenD_PE_L")
    model.eval().to(DEVICE)

    meta = pd.read_csv(meta_csv)

    # Make lookup faster and safer than df[df["obj_id"] == ...] each time
    meta["obj_id"] = meta["obj_id"].astype(str).str.strip()
    meta_by_id = meta.set_index("obj_id", drop=False)

    result_rows = []
    missed = 0

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    video_files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in video_exts
    ]

    for video_path in tqdm(video_files, total=len(video_files)):
        obj_id = video_path.stem

        if obj_id not in meta_by_id.index:
            missed += 1
            print(f"[MISS META] {obj_id}")
            continue

        meta_row = meta_by_id.loc[obj_id]

        # If duplicate obj_id somehow exists, take the first row
        if isinstance(meta_row, pd.DataFrame):
            meta_row = meta_row.iloc[0]

        try:
            label = int(meta_row["label"])
        except Exception:
            missed += 1
            print(f"[BAD LABEL] {obj_id}")
            continue

        try:
            stats = infer_video(
                model,
                str(video_path),
                stride=10,
                max_faces=1,
            )
        except Exception as e:
            missed += 1
            print(f"[FAIL INFER] {obj_id}: {repr(e)}")
            continue

        if stats is None or "avg_p_fake" not in stats:
            missed += 1
            print(f"[FAIL EMPTY] {obj_id}")
            continue

        prob_fake = float(stats["avg_p_fake"])
        pred = int(prob_fake >= threshold)

        result_rows.append({
            "obj_id": obj_id,
            "video_path": str(video_path),
            "y_true": label,
            "y_pred": pred,
            "y_prob_fake": prob_fake,
            "target": "fake" if label == 1 else "real",
            "bucket": meta_row.get("bucket", None),
            "generator": meta_row.get("generator_attrs.generator.name", None),
            "postprocessing": meta_row.get("postprocessing.pproc.name", None),
        })

    result_df = pd.DataFrame(result_rows)

    if len(result_df) == 0:
        raise RuntimeError("No videos were successfully processed.")

    y = result_df["y_true"].to_numpy()
    preds = result_df["y_pred"].to_numpy()
    probs = result_df["y_prob_fake"].to_numpy()

    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)

    # AUC requires both classes to be present
    if len(np.unique(y)) == 2:
        auc = roc_auc_score(y, probs)
    else:
        auc = float("nan")

    print("\n=== FINAL METRICS ===")
    print(f"videos_total_folder : {len(video_files)}")
    print(f"videos_total_meta   : {len(meta)}")
    print(f"videos_ok           : {len(result_rows)}")
    print(f"videos_missed       : {missed}")
    print(f"threshold           : {threshold:.3f}")
    print(f"accuracy            : {acc:.4f}")
    print(f"f1                  : {f1:.4f}")
    print(f"auc                 : {auc:.4f}" if not np.isnan(auc) else "auc                 : nan")

    print("\nClassification report:")
    print(classification_report(
        y,
        preds,
        target_names=["real", "fake"],
        digits=4,
        zero_division=0,
    ))

    print("Confusion matrix:")
    print(confusion_matrix(y, preds))

    # ── Per-group breakdowns ──────────────────────────────────────────
    for group_col in ("bucket", "generator", "postprocessing"):
        if group_col not in result_df.columns:
            continue

        if not result_df[group_col].notna().any():
            continue

        print(f"\n=== BY {group_col.upper()} ===")

        for grp_val, grp in result_df.groupby(group_col, dropna=False):
            if len(grp) < 2:
                continue

            gy = grp["y_true"].to_numpy()
            gpred = grp["y_pred"].to_numpy()
            gprob = grp["y_prob_fake"].to_numpy()

            g_acc = accuracy_score(gy, gpred)
            g_f1 = f1_score(gy, gpred, zero_division=0)

            if len(np.unique(gy)) == 2:
                g_auc = roc_auc_score(gy, gprob)
                auc_str = f"  auc={g_auc:.4f}"
            else:
                auc_str = "  auc=nan"

            print(
                f"  {grp_val}: "
                f"n={len(grp)}  "
                f"real={(gy == 0).sum()}  "
                f"fake={(gy == 1).sum()}  "
                f"acc={g_acc:.4f}  "
                f"f1={g_f1:.4f}"
                f"{auc_str}"
            )

    # Optional: save predictions for later analysis
    out_csv = folder / "gend_predictions_with_metrics.csv"
    result_df.to_csv(out_csv, index=False)
    print(f"\nSaved predictions to: {out_csv}")

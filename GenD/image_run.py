import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path

from src.hf.modeling_gend import GenD
from src.retinaface import prepare_model
from detector import align_face


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)


@torch.no_grad()
def infer_image(
    model,
    image_path: str,
    face_detector=None,
    face_thresh: float = 0.5,
    scale: float = 1.3,
    target_size: int | None = None,
    max_faces: int | None = 1,
):
    """
    Runs GenD on one image.

    Returns:
        {
            "num_faces": int,
            "avg_p_fake": float,
            "median_p_fake": float,
            "max_p_fake": float,
        }
    """

    preproc = model.feature_extractor.preprocess

    # Important: create detector outside the loop for speed if possible.
    if face_detector is None:
        face_detector = prepare_model(face_thresh)

    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        return {
            "num_faces": 0,
            "avg_p_fake": 0.0,
            "median_p_fake": 0.0,
            "max_p_fake": 0.0,
        }

    xyxy, landmarks = face_detector.detect(image_bgr)

    if xyxy is None or len(xyxy) == 0:
        return {
            "num_faces": 0,
            "avg_p_fake": 0.0,
            "median_p_fake": 0.0,
            "max_p_fake": 0.0,
        }

    # Sort faces by bbox area, largest first
    areas = [
        (i, float((b[2] - b[0]) * (b[3] - b[1])))
        for i, b in enumerate(xyxy)
    ]
    areas.sort(key=lambda x: x[1], reverse=True)

    if max_faces is None:
        indices = [i for i, _ in areas]
    else:
        indices = [i for i, _ in areas[:max_faces]]

    p_fake_values = []

    for i in indices:
        try:
            aligned, _ = align_face(
                image_bgr,
                landmarks[i],
                scale=scale,
                target_size=(target_size, target_size) if target_size else None,
            )
        except Exception:
            continue

        img = Image.fromarray(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))

        tensor = preproc(img).unsqueeze(0).to(DEVICE)

        logits = model(tensor)
        probs = logits.softmax(dim=-1)[0].detach().cpu().numpy()

        p_fake = float(probs[1])
        p_fake_values.append(p_fake)

    return {
        "num_faces": len(p_fake_values),
        "avg_p_fake": float(np.mean(p_fake_values)) if p_fake_values else 0.0,
        "median_p_fake": float(np.median(p_fake_values)) if p_fake_values else 0.0,
        "max_p_fake": float(np.max(p_fake_values)) if p_fake_values else 0.0,
    }


if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm

    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        roc_auc_score,
        classification_report,
        confusion_matrix,
    )

    folder = Path("/mnt/tank/scratch/dstoronkin/hwei_part2/images")
    meta_csv = Path("/mnt/tank/scratch/dstoronkin/hwei_part2/79f320c1-0584-45a3-8976-699213dbc35b.csv")

    threshold = 0.5

    model = GenD.from_pretrained("yermandy/GenD_PE_L")
    model.eval().to(DEVICE)

    # Create detector once. Do NOT recreate it inside infer_image for every image.
    face_detector = prepare_model(0.5)

    meta = pd.read_csv(meta_csv)

    # Fast safe lookup by obj_id
    meta["obj_id"] = meta["obj_id"].astype(str).str.strip()
    meta_by_id = meta.set_index("obj_id", drop=False)

    result_rows = []
    missed = 0

    image_exts = {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
    }

    image_files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in image_exts
    ]

    for image_path in tqdm(image_files, total=len(image_files)):
        obj_id = image_path.stem

        if obj_id not in meta_by_id.index:
            missed += 1
            print(f"[MISS META] {obj_id}")
            continue

        meta_row = meta_by_id.loc[obj_id]

        # If duplicate obj_id exists, take first one
        if isinstance(meta_row, pd.DataFrame):
            meta_row = meta_row.iloc[0]

        try:
            label = int(meta_row["label"])
        except Exception:
            missed += 1
            print(f"[BAD LABEL] {obj_id}")
            continue

        try:
            stats = infer_image(
                model=model,
                image_path=str(image_path),
                face_detector=face_detector,
                scale=1.3,
                target_size=None,
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

        # For one face avg/median/max are equal.
        # For several faces avg is usually safer than max.
        prob_fake = float(stats["avg_p_fake"])
        pred = int(prob_fake >= threshold)

        result_rows.append({
            "obj_id": obj_id,
            "image_path": str(image_path),
            "y_true": label,
            "y_pred": pred,
            "y_prob_fake": prob_fake,
            "num_faces": int(stats.get("num_faces", 0)),
            "target": "fake" if label == 1 else "real",
            "bucket": meta_row.get("bucket", None),
            "generator": meta_row.get("generator_attrs.generator.name", None),
            "postprocessing": meta_row.get("postprocessing.pproc.name", None),
        })

    result_df = pd.DataFrame(result_rows)

    if len(result_df) == 0:
        raise RuntimeError("No images were successfully processed.")

    y = result_df["y_true"].to_numpy()
    preds = result_df["y_pred"].to_numpy()
    probs = result_df["y_prob_fake"].to_numpy()

    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, zero_division=0)

    if len(np.unique(y)) == 2:
        auc = roc_auc_score(y, probs)
    else:
        auc = float("nan")

    print("\n=== FINAL METRICS ===")
    print(f"images_total_folder : {len(image_files)}")
    print(f"images_total_meta   : {len(meta)}")
    print(f"images_ok           : {len(result_rows)}")
    print(f"images_missed       : {missed}")
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

    out_csv = folder / "gend_image_predictions_with_metrics.csv"
    result_df.to_csv(out_csv, index=False)
    print(f"\nSaved predictions to: {out_csv}")
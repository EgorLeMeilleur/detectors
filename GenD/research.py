import argparse
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from src.hf.modeling_gend import GenD
from src.retinaface import prepare_model
from detector import align_face

import warnings

warnings.filterwarnings(
    "ignore",
    message=".*copying from a non-meta parameter.*",
    category=UserWarning,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_int_list(s: str) -> List[int]:
    if s is None or s.strip() == "":
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def get_video_info(video_path: str) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return {
            "total_frames": 0,
            "source_fps": 0.0,
            "duration_s": 0.0,
        }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))

    cap.release()

    duration_s = total_frames / source_fps if source_fps > 0 else 0.0

    return {
        "total_frames": total_frames,
        "source_fps": source_fps,
        "duration_s": duration_s,
    }


def uniform_indices(total_frames: int, n_frames: int) -> List[int]:
    """
    Select n_frames uniformly across the whole video.
    This is better than taking the first n frames.
    """
    if total_frames <= 0:
        return []

    n = min(int(n_frames), total_frames)

    if n <= 0:
        return []

    idxs = np.linspace(0, total_frames - 1, n)
    idxs = np.round(idxs).astype(int)
    idxs = np.unique(idxs)

    return idxs.tolist()


def stride_indices(total_frames: int, stride: int) -> List[int]:
    if total_frames <= 0:
        return []

    stride = max(1, int(stride))
    return list(range(0, total_frames, stride))


def read_frame_at(cap: cv2.VideoCapture, frame_idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


@torch.no_grad()
def infer_video_with_indices(
    model,
    video_path: str,
    frame_indices: List[int],
    face_detector,
    scale: float = 1.3,
    target_size: Optional[int] = None,
    max_faces: Optional[int] = 1,
) -> Dict[str, Any]:
    """
    Inference on selected frame indices.

    Returns averaged probability across all detected/aligned faces.
    If max_faces=1, this is effectively one probability per valid frame.
    """

    preproc = model.feature_extractor.preprocess

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return {
            "selected_frames": len(frame_indices),
            "read_frames": 0,
            "frames_with_faces": 0,
            "num_faces": 0,
            "avg_p_fake": 0.0,
            "median_p_fake": 0.0,
            "max_p_fake": 0.0,
            "elapsed_s": 0.0,
            "read_fps": 0.0,
            "face_fps": 0.0,
        }

    p_fake_values = []
    read_frames = 0
    frames_with_faces = 0
    total_faces = 0

    t0 = time.perf_counter()

    for frame_idx in frame_indices:
        frame_bgr = read_frame_at(cap, frame_idx)

        if frame_bgr is None:
            continue

        read_frames += 1

        xyxy, landmarks = face_detector.detect(frame_bgr)

        if xyxy is None or len(xyxy) == 0:
            continue

        areas = [
            (i, float((b[2] - b[0]) * (b[3] - b[1])))
            for i, b in enumerate(xyxy)
        ]
        areas.sort(key=lambda x: x[1], reverse=True)

        if max_faces is None:
            indices = [i for i, _ in areas]
        else:
            indices = [i for i, _ in areas[:max_faces]]

        frame_had_valid_face = False

        for i in indices:
            try:
                aligned, _ = align_face(
                    frame_bgr,
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

            total_faces += 1
            frame_had_valid_face = True

        if frame_had_valid_face:
            frames_with_faces += 1

    elapsed_s = time.perf_counter() - t0
    cap.release()

    read_fps = read_frames / elapsed_s if elapsed_s > 0 else 0.0
    face_fps = frames_with_faces / elapsed_s if elapsed_s > 0 else 0.0

    return {
        "selected_frames": len(frame_indices),
        "read_frames": read_frames,
        "frames_with_faces": frames_with_faces,
        "num_faces": total_faces,
        "avg_p_fake": float(np.mean(p_fake_values)) if p_fake_values else 0.0,
        "median_p_fake": float(np.median(p_fake_values)) if p_fake_values else 0.0,
        "max_p_fake": float(np.max(p_fake_values)) if p_fake_values else 0.0,
        "elapsed_s": float(elapsed_s),
        "read_fps": float(read_fps),
        "face_fps": float(face_fps),
    }


def compute_dataset_metrics(
    result_df: pd.DataFrame,
    threshold: float,
) -> Dict[str, Any]:
    y = result_df["y_true"].to_numpy()
    probs = result_df["y_prob_fake"].to_numpy()
    preds = (probs >= threshold).astype(int)

    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, zero_division=0)

    if len(np.unique(y)) == 2:
        auc = roc_auc_score(y, probs)
    else:
        auc = float("nan")

    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "auc": float(auc),
    }


def make_run_configs(
    strides: List[int],
    frame_counts: List[int],
) -> List[Dict[str, Any]]:
    configs = []

    for stride in strides:
        configs.append({
            "run_name": f"stride_{stride}",
            "mode": "stride",
            "stride": int(stride),
            "n_frames": None,
        })

    for n in frame_counts:
        configs.append({
            "run_name": f"uniform_{n}_frames",
            "mode": "uniform",
            "stride": None,
            "n_frames": int(n),
        })

    return configs


def collect_video_files(folder: Path, recursive: bool = False) -> List[Path]:
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    if recursive:
        files = [
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in video_exts
        ]
    else:
        files = [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in video_exts
        ]

    return sorted(files)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--folder",
        type=str,
        default="/mnt/tank/scratch/dstoronkin/hwei_part1",
        help="Folder with videos and metadata CSV.",
    )
    parser.add_argument(
        "--meta-csv",
        default="/mnt/tank/scratch/dstoronkin/hwei_part1/d83a0ce6-dc87-46a6-9679-98a71cf91886.csv",
        help="CSV with obj_id,label,bucket,generator_attrs.generator.name,etc.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="research_out",
        help="Output directory. Default: folder/frame_sampling_eval",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="yermandy/GenD_PE_L",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--strides",
        type=str,
        default="1,2,4,5,10,15,20,30",
        help="Comma-separated stride values. stride=1 means all frames.",
    )
    parser.add_argument(
        "--frame-counts",
        type=str,
        default="100,64,32,16",
        help="Comma-separated fixed number of uniformly sampled frames.",
    )

    parser.add_argument(
        "--face-thresh",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.3,
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=1,
        help="Use 1 for largest face only. Use 0 for all faces.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search videos recursively.",
    )

    args = parser.parse_args()

    folder = Path(args.folder)
    meta_csv = Path(args.meta_csv)

    out_dir = Path(args.out_dir) if args.out_dir else folder / "frame_sampling_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    max_faces = None if args.max_faces == 0 else args.max_faces

    strides = parse_int_list(args.strides)
    frame_counts = parse_int_list(args.frame_counts)
    run_configs = make_run_configs(strides, frame_counts)

    print(f"DEVICE: {DEVICE}")
    print(f"Folder: {folder}")
    print(f"Meta CSV: {meta_csv}")
    print(f"Output dir: {out_dir}")
    print(f"Runs: {[c['run_name'] for c in run_configs]}")

    meta = pd.read_csv(meta_csv)
    meta["obj_id"] = meta["obj_id"].astype(str).str.strip()
    meta_by_id = meta.set_index("obj_id", drop=False)

    video_files = collect_video_files(folder, recursive=args.recursive)

    print(f"Found videos: {len(video_files)}")
    print(f"Metadata rows: {len(meta)}")

    model = GenD.from_pretrained(args.model_name)
    model.eval().to(DEVICE)

    # Important: create RetinaFace once.
    face_detector = prepare_model(args.face_thresh)

    summary_rows = []

    for config in run_configs:
        run_name = config["run_name"]
        mode = config["mode"]

        print(f"\n==============================")
        print(f"RUN: {run_name}")
        print(f"==============================")

        result_rows = []
        missed = 0

        run_t0 = time.perf_counter()

        for video_path in tqdm(video_files, total=len(video_files)):
            obj_id = video_path.stem

            if obj_id not in meta_by_id.index:
                missed += 1
                print(f"[MISS META] {obj_id}")
                continue

            meta_row = meta_by_id.loc[obj_id]

            if isinstance(meta_row, pd.DataFrame):
                meta_row = meta_row.iloc[0]

            try:
                label = int(meta_row["label"])
            except Exception:
                missed += 1
                print(f"[BAD LABEL] {obj_id}")
                continue

            info = get_video_info(str(video_path))
            total_frames = int(info["total_frames"])

            if total_frames <= 0:
                missed += 1
                print(f"[BAD VIDEO] {obj_id}")
                continue

            if mode == "stride":
                frame_indices = stride_indices(
                    total_frames=total_frames,
                    stride=config["stride"],
                )
            elif mode == "uniform":
                frame_indices = uniform_indices(
                    total_frames=total_frames,
                    n_frames=config["n_frames"],
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")

            if len(frame_indices) == 0:
                missed += 1
                print(f"[NO FRAMES] {obj_id}")
                continue

            try:
                stats = infer_video_with_indices(
                    model=model,
                    video_path=str(video_path),
                    frame_indices=frame_indices,
                    face_detector=face_detector,
                    scale=args.scale,
                    target_size=args.target_size,
                    max_faces=max_faces,
                )
            except Exception as e:
                missed += 1
                print(f"[FAIL INFER] {obj_id}: {repr(e)}")
                continue

            prob_fake = float(stats["avg_p_fake"])
            pred = int(prob_fake >= args.threshold)

            result_rows.append({
                "run_name": run_name,
                "mode": mode,
                "stride": config["stride"],
                "n_frames_requested": config["n_frames"],

                "obj_id": obj_id,
                "video_path": str(video_path),

                "y_true": label,
                "y_pred": pred,
                "y_prob_fake": prob_fake,

                "target": "fake" if label == 1 else "real",
                "bucket": meta_row.get("bucket", None),
                "generator": meta_row.get("generator_attrs.generator.name", None),
                "postprocessing": meta_row.get("postprocessing.pproc.name", None),

                "video_total_frames": total_frames,
                "video_source_fps": info["source_fps"],
                "video_duration_s": info["duration_s"],

                "selected_frames": stats["selected_frames"],
                "read_frames": stats["read_frames"],
                "frames_with_faces": stats["frames_with_faces"],
                "num_faces": stats["num_faces"],

                "median_p_fake": stats["median_p_fake"],
                "max_p_fake": stats["max_p_fake"],

                "elapsed_s": stats["elapsed_s"],
                "read_fps": stats["read_fps"],
                "face_fps": stats["face_fps"],
            })

        run_elapsed_s = time.perf_counter() - run_t0

        result_df = pd.DataFrame(result_rows)

        if len(result_df) == 0:
            print(f"[WARNING] No successful videos for {run_name}")
            continue

        metrics = compute_dataset_metrics(result_df, args.threshold)

        y = result_df["y_true"].to_numpy()
        probs = result_df["y_prob_fake"].to_numpy()
        preds = (probs >= args.threshold).astype(int)

        print("\n=== FINAL METRICS ===")
        print(f"run_name            : {run_name}")
        print(f"mode                : {mode}")
        print(f"stride              : {config['stride']}")
        print(f"n_frames_requested  : {config['n_frames']}")
        print(f"videos_total_folder : {len(video_files)}")
        print(f"videos_total_meta   : {len(meta)}")
        print(f"videos_ok           : {len(result_rows)}")
        print(f"videos_missed       : {missed}")
        print(f"threshold           : {args.threshold:.3f}")
        print(f"accuracy            : {metrics['accuracy']:.4f}")
        print(f"f1                  : {metrics['f1']:.4f}")
        print(
            f"auc                 : {metrics['auc']:.4f}"
            if not np.isnan(metrics["auc"])
            else "auc                 : nan"
        )

        print(f"run_elapsed_s       : {run_elapsed_s:.2f}")
        print(f"avg_read_fps        : {result_df['read_fps'].mean():.2f}")
        print(f"avg_face_fps        : {result_df['face_fps'].mean():.2f}")
        print(f"avg_selected_frames : {result_df['selected_frames'].mean():.2f}")
        print(f"avg_frames_w_faces  : {result_df['frames_with_faces'].mean():.2f}")

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

        pred_csv = out_dir / f"predictions_{run_name}.csv"
        result_df.to_csv(pred_csv, index=False)
        print(f"\nSaved predictions to: {pred_csv}")

        summary_rows.append({
            "run_name": run_name,
            "mode": mode,
            "stride": config["stride"],
            "n_frames_requested": config["n_frames"],

            "videos_total_folder": len(video_files),
            "videos_total_meta": len(meta),
            "videos_ok": len(result_rows),
            "videos_missed": missed,

            "threshold": args.threshold,
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "auc": metrics["auc"],

            "run_elapsed_s": run_elapsed_s,
            "avg_read_fps": float(result_df["read_fps"].mean()),
            "avg_face_fps": float(result_df["face_fps"].mean()),
            "avg_selected_frames": float(result_df["selected_frames"].mean()),
            "avg_read_frames": float(result_df["read_frames"].mean()),
            "avg_frames_with_faces": float(result_df["frames_with_faces"].mean()),
            "avg_num_faces": float(result_df["num_faces"].mean()),
            "avg_video_total_frames": float(result_df["video_total_frames"].mean()),
            "avg_prob_fake": float(result_df["y_prob_fake"].mean()),
        })

    summary_df = pd.DataFrame(summary_rows)

    if len(summary_df) > 0:
        summary_csv = out_dir / "summary_metrics.csv"
        summary_df.to_csv(summary_csv, index=False)

        print("\n==============================")
        print("ALL RUNS SUMMARY")
        print("==============================")
        print(summary_df.sort_values("auc", ascending=False).to_string(index=False))
        print(f"\nSaved summary to: {summary_csv}")


if __name__ == "__main__":
    main()
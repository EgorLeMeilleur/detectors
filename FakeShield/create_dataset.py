#!/usr/bin/env python3
"""Extract first video frames and build DTE-FDM LoRA training JSON.

Expected input layout:

video_root/
  real/
    *.mp4
  fake/
    method_a/
      *.mp4
    method_b/
      *.mp4

Example:
python build_dte_json.py \
  --video-root /path/to/videos \
  --frame-root /path/to/frames \
  --output /path/to/train.json \
  --json-image-root /path/to/frames \
  --skip-bad-videos
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable


BASE_PROMPT = (
    "Was this photo taken directly from the camera without any processing? "
    "Has it been tampered with by any artificial photo modification techniques such as ps? "
    "Please zoom in on any details in the image, paying special attention to the edges "
    "of the objects, capturing some unnatural edges and perspective relationships, "
    "some incorrect semantics, unnatural lighting and darkness etc."
)

DOMAIN_PREFIXES = {
    "none": "",
    "aigc": "This is a picture that is suspected to have been tampered with by AIGC inpainting. ",
    "aigc_inpainting": "This is a picture that is suspected to have been tampered with by AIGC inpainting. ",
    "deepfake": "This is a picture that is suspected to have been tampered with by DeepFake. ",
    "photoshop": "This is a picture that is suspected to have been tampered with by Photoshop. ",
    "ps": "This is a picture that is suspected to have been tampered with by Photoshop. ",
}

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def normalize_part(value: str) -> str:
    return str(value).strip().lower()


def build_prompt(domain: str) -> str:
    prefix = DOMAIN_PREFIXES.get(normalize_part(domain), "")
    return "<image>\n" + prefix + BASE_PROMPT


def real_answer() -> str:
    return (
        "1. Whether the picture has been tampered with / Description of the tampered area: "
        "The picture does not appear to have been tampered with.\n\n"
        "2. Judgment basis: The lighting, shadows, object boundaries, perspective, texture "
        "resolution, and semantic content are generally consistent across the image. No clear "
        "artificial modification artifacts are visible."
    )


def fake_answer(method: str) -> str:
    method_text = method.replace("_", " ").replace("-", " ").strip()

    if method_text:
        method_sentence = f" The sample belongs to the fake subset '{method_text}'."
    else:
        method_sentence = ""

    return (
        "1. Whether the picture has been tampered with / Description of the tampered area: "
        "The picture has been tampered with. The manipulated content is expected to appear "
        "mainly in the face or identity-related region of the frame."
        f"{method_sentence}\n\n"
        "2. Judgment basis: The image should be inspected for forgery traces such as unnatural "
        "facial boundaries, inconsistent skin texture, abnormal lighting, mismatched shadows, "
        "warped facial geometry, local blur, compression differences, and semantic inconsistencies "
        "between the manipulated region and the surrounding scene."
    )


def iter_videos(video_root: Path) -> Iterable[tuple[Path, str, str]]:
    real_root = video_root / "real"

    if real_root.exists():
        for path in sorted(real_root.rglob("*")):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                yield path, "real", ""

    fake_root = video_root / "fake"

    if fake_root.exists():
        for path in sorted(fake_root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue

            rel_parts = path.relative_to(fake_root).parts
            method = rel_parts[0] if len(rel_parts) > 1 else ""
            yield path, "fake", method


def frame_path_for(video_path: Path, video_root: Path, frame_root: Path) -> Path:
    rel_path = video_path.relative_to(video_root)
    return frame_root / rel_path.with_suffix(".jpg")


def extract_first_frame(
    video_path: Path,
    frame_path: Path,
    *,
    overwrite: bool = False,
    jpeg_quality: int = 95,
    max_read_attempts: int = 30,
) -> str:
    """Extract first decodable frame.

    Returns:
        "exists"  - frame already exists
        "written" - frame was written
        "skipped" - video could not be decoded or written
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required. Install it with: pip install opencv-python"
        ) from exc

    if frame_path.exists() and not overwrite:
        return "exists"

    frame_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return "skipped"

        frame = None

        # Try frame 0 first.
        ok, candidate = capture.read()
        if ok and candidate is not None:
            frame = candidate
        else:
            # Some videos fail on frame 0 but decode after a few reads.
            for _ in range(max_read_attempts):
                ok, candidate = capture.read()
                if ok and candidate is not None:
                    frame = candidate
                    break

        if frame is None:
            return "skipped"

        success = cv2.imwrite(
            str(frame_path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )

        if not success:
            return "skipped"

        return "written"

    finally:
        capture.release()


def json_image_path(frame_path: Path, json_image_root: Path | None) -> str:
    if json_image_root:
        return os.path.relpath(frame_path, json_image_root).replace(os.sep, "/")
    return str(frame_path).replace(os.sep, "/")


def build_example(
    sample_id: str,
    image_path: str,
    label: str,
    method: str,
    fake_domain: str,
) -> dict:
    is_fake = label == "fake"
    domain = fake_domain if is_fake else "none"
    answer = fake_answer(method) if is_fake else real_answer()

    return {
        "id": sample_id,
        "image": image_path,
        "label": label,
        "fake_method": method,
        "conversations": [
            {"from": "human", "value": build_prompt(domain)},
            {"from": "gpt", "value": answer},
        ],
    }


def convert(args: argparse.Namespace) -> None:
    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise ImportError(
            "tqdm is required for the progress bar. Install it with: pip install tqdm"
        ) from exc

    video_root = Path(args.video_root).resolve()
    frame_root = Path(args.frame_root).resolve()
    json_image_root = Path(args.json_image_root).resolve() if args.json_image_root else None

    videos = list(iter_videos(video_root))

    if args.limit is not None:
        videos = videos[: args.limit]

    examples = []
    skipped = 0
    written = 0
    existing = 0

    pbar = tqdm(videos, total=len(videos), desc="Extracting first frames", unit="video")

    for index, (video_path, label, method) in enumerate(pbar):
        frame_path = frame_path_for(video_path, video_root, frame_root)

        try:
            status = extract_first_frame(
                video_path,
                frame_path,
                overwrite=args.overwrite_frames,
                jpeg_quality=args.jpeg_quality,
                max_read_attempts=args.max_read_attempts,
            )
        except Exception as exc:
            if not args.skip_bad_videos:
                raise

            skipped += 1
            pbar.write(f"[SKIP] {video_path}: {exc}")
            continue

        if status == "written":
            written += 1
        elif status == "exists":
            existing += 1
        else:
            skipped += 1
            if args.verbose_skips:
                pbar.write(f"[SKIP] Could not decode/write: {video_path}")
            continue

        image_path = json_image_path(frame_path, json_image_root)
        sample_id = f"{label}_{index:06d}"

        examples.append(
            build_example(
                sample_id=sample_id,
                image_path=image_path,
                label=label,
                method=method,
                fake_domain=args.fake_domain,
            )
        )

        pbar.set_postfix(
            written=written,
            reused=existing,
            skipped=skipped,
            examples=len(examples),
        )

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(examples)} training examples to {output}")
    print(f"Frames written: {written}; reused: {existing}; skipped videos: {skipped}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a first-frame image dataset from real/fake videos and build "
            "DTE-FDM LLaVA-style LoRA training JSON."
        )
    )

    parser.add_argument("--video-root", required=True, help="Root containing real/ and fake/ folders.")
    parser.add_argument("--frame-root", required=True, help="Output root for extracted first-frame JPGs.")
    parser.add_argument("--output", required=True, help="Output DTE-FDM training JSON path.")

    parser.add_argument(
        "--json-image-root",
        default="",
        help=(
            "If set, image paths in JSON are relative to this folder. "
            "Use the same folder as finetune_lora.sh --image_folder."
        ),
    )

    parser.add_argument(
        "--fake-domain",
        default="deepfake",
        choices=sorted(DOMAIN_PREFIXES.keys()),
        help="Domain prompt prefix used for fake samples.",
    )

    parser.add_argument(
        "--overwrite-frames",
        action="store_true",
        help="Re-extract existing JPG frames.",
    )

    parser.add_argument(
        "--skip-bad-videos",
        action="store_true",
        help="Skip unreadable videos instead of failing.",
    )

    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        choices=range(1, 101),
        metavar="[1-100]",
    )

    parser.add_argument(
        "--max-read-attempts",
        type=int,
        default=30,
        help=(
            "Maximum number of frame reads before treating a video as bad. "
            "Prevents infinite loops on broken videos."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N videos. Useful for debugging.",
    )

    parser.add_argument(
        "--verbose-skips",
        action="store_true",
        help="Print every skipped video.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
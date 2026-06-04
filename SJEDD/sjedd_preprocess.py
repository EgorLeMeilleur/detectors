"""
Universal SJEDD Preprocessing Script
======================================
Replicates the exact face extraction pipeline used in SJEDD training/testing:
  1. Read video frames (or load images directly)
  2. Detect faces with RetinaFace
  3. Select the largest face per frame
  4. Landmark-align and crop to 317×317 using norm_crop
  5. Save as PNG

Input structure (videos):
    input_root/
    ├── fake/
    │   ├── video1.mp4
    │   ├── video2.avi
    │   └── ...
    └── real/
        ├── video1.mp4
        └── ...

Input structure (images — already face images, just need alignment):
    input_root/
    ├── fake/
    │   ├── img001.jpg
    │   └── ...
    └── real/
        └── ...

Output structure (always):
    output_root/
    ├── fake/
    │   ├── video1/          # one sub-folder per source video/image
    │   │   ├── 0000.png
    │   │   ├── 0001.png
    │   │   └── ...
    │   └── video2/
    └── real/
        └── ...

This output structure feeds directly into sjedd_universal_benchmark.py --mode video.
For image-mode benchmarking you can also use --mode image by pointing at a flat folder,
but video mode (averaging frames per video) is what the original SJEDD papers report.

Requirements:
    pip install opencv-python-headless numpy tqdm
    pip install torch torchvision   # for RetinaFace detector

    You also need:
      - face_utils2.py  (from SJEDD /preprocessing folder)
      - RetinaFace-Resnet50-fixed.pth  (checkpoint for face detector)

    Both are available in the SJEDD repo's preprocessing/ folder.
    Put this script next to face_utils2.py, or set --face_utils_dir.

Usage:
    # Video input
    python sjedd_preprocess.py \\
        --input_root /data/my_raw_dataset \\
        --output_root /data/my_preprocessed \\
        --input_type video

    # Image input (e.g. already extracted frames, or still images)
    python sjedd_preprocess.py \\
        --input_root /data/my_images \\
        --output_root /data/my_preprocessed \\
        --input_type image

    # Only process every Nth frame (to reduce dataset size)
    python sjedd_preprocess.py \\
        --input_root /data/my_raw_dataset \\
        --output_root /data/my_preprocessed \\
        --input_type video \\
        --frame_skip 5

    # Use a different detector checkpoint path
    python sjedd_preprocess.py \\
        --input_root /data/my_raw_dataset \\
        --output_root /data/my_preprocessed \\
        --detector_ckpt ./preprocessing/RetinaFace-Resnet50-fixed.pth
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from tqdm import tqdm

import cv2
import numpy as np

# ── Try importing face_utils2 (from SJEDD preprocessing folder) ───────────────
def _import_face_utils(face_utils_dir: str = None):
    if face_utils_dir:
        sys.path.insert(0, face_utils_dir)
    # Also try ./preprocessing relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for candidate in [script_dir, os.path.join(script_dir, 'preprocessing')]:
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    try:
        from face_utils2 import FaceDetector, norm_crop
        return FaceDetector, norm_crop
    except ImportError as e:
        print(e)
        print(
            "[ERROR] Cannot import face_utils2.\n"
            "  Make sure face_utils2.py is in the same directory as this script,\n"
            "  or pass --face_utils_dir /path/to/sjedd/preprocessing\n"
            "  (it lives in the SJEDD repo under preprocessing/)"
        )
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# Core extraction helpers
# ══════════════════════════════════════════════════════════════════════════════

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.m4v', '.webm'}

OUTSIZE = (317, 317)  # exact size used in original SJEDD preprocessing


def make_face_detector(detector_ckpt: str, FaceDetector):
    """Create and cache a single FaceDetector instance."""
    fd = FaceDetector()
    fd.load_checkpoint(detector_ckpt)
    return fd


def extract_face_from_frame(bgr_frame, face_detector, norm_crop, source_desc='', frame_num=0,
                             detect_scale=0.2):  # add this param
    h, w = bgr_frame.shape[:2]

    if detect_scale != 1.0:
        small = cv2.resize(bgr_frame, (int(w*detect_scale), int(h*detect_scale)))
    else:
        small = bgr_frame

    try:
        boxes, landms = face_detector.detect(small)
    except AttributeError:
        return None

    if boxes is None or boxes.shape[0] == 0:
        return None

    # Scale landmarks back up to original resolution for norm_crop
    if detect_scale != 1.0:
        boxes = boxes / detect_scale
        landms = landms / detect_scale

    areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    max_idx = int(areas.argmax())
    landm = landms[max_idx]

    try:
        landmarks = landm.detach().numpy().reshape(5, 2).astype(np.int32)
    except Exception:
        landmarks = np.array(landm).reshape(5, 2).astype(np.int32)

    aligned = norm_crop(bgr_frame, landmarks, outsize=OUTSIZE)  # crop from ORIGINAL
    return aligned

def process_video(video_path: str, output_folder: str, face_detector, norm_crop,
                  frame_skip: int = 1, max_frames: int = None):
    """
    Extract all frames from a video, detect+align faces, save as 0000.png, 0001.png …
    Skips frames where no face is detected (matching original SJEDD behaviour).

    Args:
        video_path:    Path to source video file.
        output_folder: Directory to write face PNG crops into.
        frame_skip:    Save only every Nth frame (1 = every frame, 5 = every 5th).
        max_frames:    Stop after saving this many face crops (None = no limit).
    """
    os.makedirs(output_folder, exist_ok=True)

    reader = cv2.VideoCapture(video_path)
    if not reader.isOpened():
        print(f"[WARNING] Cannot open video: {video_path}")
        return 0

    frame_num = 0       # raw frame counter
    saved_count = 0     # how many crops written

    while reader.isOpened():
        success, bgr = reader.read()
        if not success:
            break

        out_path = os.path.join(output_folder, f"{saved_count:04d}.png")
        cv2.imwrite(out_path, bgr)
        saved_count += 1
        if max_frames and saved_count >= max_frames:
            break

        # if frame_num % frame_skip == 0:
        #     face = extract_face_from_frame(
        #         bgr, face_detector, norm_crop,
        #         source_desc=video_path, frame_num=frame_num,
        #         detect_scale=1.0
        #     )

        #     if face is not None:
        #         out_path = os.path.join(output_folder, f"{saved_count:04d}.png")
        #         cv2.imwrite(out_path, face)
        #         saved_count += 1
        #         if max_frames and saved_count >= max_frames:
        #             break

        frame_num += 1

    reader.release()
    return saved_count


def process_image(image_path: str, output_folder: str, face_detector, norm_crop,
                  image_index: int = 0):
    """
    Load a single image, detect+align the largest face, save as NNNN.png.
    Used when input_type='image' (flat folder of images, one sample per image).
    Each image gets its own sub-folder so the output is always video-mode compatible.
    """
    os.makedirs(output_folder, exist_ok=True)

    bgr = cv2.imread(image_path)
    if bgr is None:
        print(f"[WARNING] Cannot read image: {image_path}")
        return 0

    face = extract_face_from_frame(
        bgr, face_detector, norm_crop,
        source_desc=image_path, frame_num=0
    )
    if face is None:
        print(f"[WARNING] No face detected: {image_path}")
        return 0

    out_path = os.path.join(output_folder, f"{image_index:04d}.png")
    cv2.imwrite(out_path, face)
    return 1


# ══════════════════════════════════════════════════════════════════════════════
# High-level dataset walkers
# ══════════════════════════════════════════════════════════════════════════════

def process_video_dataset(input_root: str, output_root: str, face_detector, norm_crop,
                          fake_label: str, real_label: str,
                          frame_skip: int, max_frames: int):
    """Walk fake/ and real/ sub-folders, treating each file as a video."""
    stats = {'processed': 0, 'skipped': 0, 'total_faces': 0}

    for class_label in [real_label, fake_label]:
        input_class_dir = os.path.join(input_root, class_label)
        output_class_dir = os.path.join(output_root, class_label)

        if not os.path.isdir(input_class_dir):
            print(f"[WARNING] Input folder not found, skipping: {input_class_dir}")
            continue

        video_files = sorted([
            f for f in os.listdir(input_class_dir)
            if Path(f).suffix.lower() in VIDEO_EXTENSIONS
        ])

        if len(video_files) == 0:
            print(f"[WARNING] No video files found in {input_class_dir}")
            print(f"          Supported extensions: {VIDEO_EXTENSIONS}")
            continue

        print(f"\n[{class_label.upper()}] Processing {len(video_files)} videos → {output_class_dir}")

        for video_file in tqdm(video_files, desc=class_label):
            video_path = os.path.join(input_class_dir, video_file)
            video_stem = Path(video_file).stem
            output_folder = os.path.join(output_class_dir, video_stem)

            # Skip if already processed (allows resuming interrupted runs)
            if os.path.isdir(output_folder) and len(os.listdir(output_folder)) > 0:
                stats['skipped'] += 1
                continue

            n = process_video(video_path, output_folder, face_detector, norm_crop,
                              frame_skip=frame_skip, max_frames=max_frames)
            stats['processed'] += 1
            stats['total_faces'] += n

    return stats


def process_image_dataset(input_root: str, output_root: str, face_detector, norm_crop,
                          fake_label: str, real_label: str):
    """
    Walk fake/ and real/ sub-folders treating each file as a standalone image.
    Output: one sub-folder per image (so benchmark can run in video mode averaging frames,
    though for images it's just 1 frame per 'video').
    Also writes a flat copy so benchmark --mode image works directly.
    """
    stats = {'processed': 0, 'skipped': 0, 'total_faces': 0}

    for class_label in [real_label, fake_label]:
        input_class_dir = os.path.join(input_root, class_label)
        output_class_dir = os.path.join(output_root, class_label)

        if not os.path.isdir(input_class_dir):
            print(f"[WARNING] Input folder not found, skipping: {input_class_dir}")
            continue

        image_files = sorted([
            f for f in os.listdir(input_class_dir)
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS
        ])

        if len(image_files) == 0:
            print(f"[WARNING] No image files found in {input_class_dir}")
            continue

        print(f"\n[{class_label.upper()}] Processing {len(image_files)} images → {output_class_dir}")
        os.makedirs(output_class_dir, exist_ok=True)

        for img_file in tqdm(image_files, desc=class_label):
            img_path = os.path.join(input_class_dir, img_file)
            img_stem = Path(img_file).stem

            # Each image → its own sub-folder (video-mode compatible)
            subfolder = os.path.join(output_class_dir, img_stem)
            if os.path.isdir(subfolder) and len(os.listdir(subfolder)) > 0:
                stats['skipped'] += 1
                continue

            n = process_image(img_path, subfolder, face_detector, norm_crop, image_index=0)
            stats['processed'] += 1
            stats['total_faces'] += n

    return stats


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description='Universal SJEDD face preprocessing — RetinaFace + norm_crop → 317×317',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--input_root', required=True,
                        help='Root folder containing fake/ and real/ sub-folders')
    parser.add_argument('--output_root', required=True,
                        help='Where to write processed face crops')
    parser.add_argument('--input_type', default='video', choices=['video', 'image'],
                        help='"video": each file is a video to extract frames from. '
                             '"image": each file is already a single image. (default: video)')
    parser.add_argument('--detector_ckpt', default='RetinaFace-Resnet50-fixed.pth',
                        help='Path to RetinaFace checkpoint (default: RetinaFace-Resnet50-fixed.pth)')
    parser.add_argument('--face_utils_dir', default=None,
                        help='Directory containing face_utils2.py if not in current dir '
                             '(e.g. /path/to/SJEDD/preprocessing)')
    parser.add_argument('--fake_label', default='crop_img',
                        help='Sub-folder name for fake class (default: fake)')
    parser.add_argument('--real_label', default='real',
                        help='Sub-folder name for real class (default: real)')
    parser.add_argument('--frame_skip', type=int, default=1,
                        help='[Video mode] Save every Nth frame. 1=every frame (default), '
                             '5=every 5th. Useful to reduce storage without losing much.')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='[Video mode] Max face crops to save per video. '
                             'None=no limit (default). Use e.g. 300 to cap long videos.')
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Validate paths ─────────────────────────────────────────────────────────
    if not os.path.isdir(args.input_root):
        sys.exit(f"[ERROR] input_root not found: {args.input_root}")

    if not os.path.isfile(args.detector_ckpt):
        sys.exit(
            f"[ERROR] Detector checkpoint not found: {args.detector_ckpt}\n"
            f"  Download RetinaFace-Resnet50-fixed.pth from the SJEDD preprocessing/ folder\n"
            f"  or pass --detector_ckpt /full/path/to/RetinaFace-Resnet50-fixed.pth"
        )

    # ── Load face_utils2 ───────────────────────────────────────────────────────
    FaceDetector, norm_crop = _import_face_utils(args.face_utils_dir)

    # ── Build detector (created once, reused for all frames) ──────────────────
    print(f"[Detector] Loading RetinaFace from {args.detector_ckpt}")
    face_detector = make_face_detector(args.detector_ckpt, FaceDetector)

    print(f"[Config] input_type={args.input_type} | "
          f"outsize={OUTSIZE} | "
          f"frame_skip={args.frame_skip} | "
          f"max_frames={args.max_frames}")
    print(f"[Input]  {args.input_root}")
    print(f"[Output] {args.output_root}")

    # ── Process ────────────────────────────────────────────────────────────────
    if args.input_type == 'video':
        stats = process_video_dataset(
            args.input_root, args.output_root,
            face_detector, norm_crop,
            fake_label=args.fake_label,
            real_label=args.real_label,
            frame_skip=args.frame_skip,
            max_frames=args.max_frames,
        )
    else:
        stats = process_image_dataset(
            args.input_root, args.output_root,
            face_detector, norm_crop,
            fake_label=args.fake_label,
            real_label=args.real_label,
        )

    # ── Summary ────────────────────────────────────────────────────────────────
    print('\n' + '='*50)
    print(f"  Done!")
    print(f"  Processed : {stats['processed']}")
    print(f"  Skipped   : {stats['skipped']}  (already existed)")
    print(f"  Face crops: {stats['total_faces']}")
    print(f"  Output    : {args.output_root}")
    print('='*50)
    print()
    print("Next step — run the benchmark:")
    print(f"  python sjedd_universal_benchmark.py \\")
    print(f"      --datapath {args.output_root} \\")
    print(f"      --resume ./pretrained/ckpt_best.pth \\")
    print(f"      --mode video \\")
    print(f"      --n_frames 32")


if __name__ == '__main__':
    main()

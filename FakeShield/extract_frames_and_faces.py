import os
import cv2
import argparse
from tqdm import tqdm
import mediapipe as mp


# -----------------------------
# Extract N frames with stride
# -----------------------------
def extract_frames(video_path, num_frames=15, stride=5, start_frame=0):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return []

    target_indices = [start_frame + i * stride for i in range(num_frames)]
    target_set = set(target_indices)

    frames = []
    frame_idx = 0

    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in target_set:
            frames.append((frame_idx, frame))

        frame_idx += 1

        # Small optimization: stop after the last requested frame
        if frame_idx > target_indices[-1]:
            break

    cap.release()
    return frames


# -----------------------------
# Face detection + padded crop
# -----------------------------
mp_face = mp.solutions.face_detection


def crop_face_with_padding(detector, image, padding_ratio=0.3):
    h, w, _ = image.shape

    results = detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.detections:
        return None

    bbox = results.detections[0].location_data.relative_bounding_box

    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    bw = int(bbox.width * w)
    bh = int(bbox.height * h)

    pad_x = int(bw * padding_ratio)
    pad_y = int(bh * padding_ratio)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    return image[y1:y2, x1:x2]


# -----------------------------
# Main
# -----------------------------
def process_folder(
    input_folder,
    output_full,
    output_face=None,
    num_frames=15,
    stride=5,
    start_frame=0,
    jpeg_quality=95,
):
    os.makedirs(output_full, exist_ok=True)

    if output_face:
        os.makedirs(output_face, exist_ok=True)

    video_files = [
        f for f in sorted(os.listdir(input_folder))
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"))
    ]

    detector = None
    if output_face:
        detector = mp_face.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5,
        )

    try:
        for video_name in tqdm(video_files, desc="Processing videos", unit="video"):
            video_path = os.path.join(input_folder, video_name)
            video_stem = os.path.splitext(video_name)[0]

            frames = extract_frames(
                video_path,
                num_frames=num_frames,
                stride=stride,
                start_frame=start_frame,
            )

            if not frames:
                continue

            for frame_idx, frame in frames:
                image_name = f"{video_stem}_frame{frame_idx:06d}.jpg"

                full_out_path = os.path.join(output_full, image_name)
                cv2.imwrite(
                    full_out_path,
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
                )

                if output_face:
                    face_crop = crop_face_with_padding(detector, frame)
                    if face_crop is not None:
                        face_out_path = os.path.join(output_face, image_name)
                        cv2.imwrite(
                            face_out_path,
                            face_crop,
                            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
                        )

    finally:
        if detector is not None:
            detector.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-folder",
        type=str,
        default="/mnt/tank/scratch/dstoronkin/hwei_part1",
    )

    parser.add_argument(
        "--output-full",
        type=str,
        default="/mnt/tank/scratch/dstoronkin/fakeshield_test/hwei_part1_15_stride5_full",
    )

    parser.add_argument(
        "--output-face",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--num-frames",
        type=int,
        default=15,
        help="Number of frames to save per video.",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Frame stride.",
    )

    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First frame index.",
    )

    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        choices=range(1, 101),
        metavar="[1-100]",
    )

    args = parser.parse_args()

    process_folder(
        input_folder=args.input_folder,
        output_full=args.output_full,
        output_face=args.output_face,
        num_frames=args.num_frames,
        stride=args.stride,
        start_frame=args.start_frame,
        jpeg_quality=args.jpeg_quality,
    )
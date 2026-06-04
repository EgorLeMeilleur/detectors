import os
from pathlib import Path
from collections import defaultdict

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

FAKE_KEYWORDS = [
    "gan", "swap", "fake", "deepfake", "style", "diff",
    "sadtalker", "wav2lip", "faceswap", "fsgan"
]

def infer_label(name: str):
    name = name.lower()
    if "real" in name:
        return "real"
    if any(k in name for k in FAKE_KEYWORDS):
        return "fake"
    return "unknown"


def analyze_folder(root: Path):
    stats = {
        "images": 0,
        "videos": 0,
        "subfolders": 0,
        "folders_with_images": 0,
        "real_dirs": 0,
        "fake_dirs": 0,
    }

    folder_image_counts = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirpath = Path(dirpath)

        stats["subfolders"] += len(dirnames)

        imgs = [f for f in filenames if Path(f).suffix.lower() in IMAGE_EXT]
        vids = [f for f in filenames if Path(f).suffix.lower() in VIDEO_EXT]

        stats["images"] += len(imgs)
        stats["videos"] += len(vids)

        if imgs:
            stats["folders_with_images"] += 1
            folder_image_counts.append(len(imgs))

        # detect real/fake folders
        for d in dirnames:
            d_lower = d.lower()
            if "real" in d_lower:
                stats["real_dirs"] += 1
            if "fake" in d_lower:
                stats["fake_dirs"] += 1

    # structure type
    if stats["real_dirs"] > 0 and stats["fake_dirs"] > 0:
        structure = "real/fake split"
    elif stats["folders_with_images"] > 10:
        structure = "many image folders (likely frames)"
    elif stats["images"] > 0:
        structure = "flat images"
    elif stats["videos"] > 0:
        structure = "videos"
    else:
        structure = "empty"

    avg_imgs = int(sum(folder_image_counts) / len(folder_image_counts)) if folder_image_counts else 0

    return {
        **stats,
        "structure": structure,
        "avg_images_per_folder": avg_imgs,
    }


def main():
    dataset_root = Path("/mnt/tank/scratch/dstoronkin/df40/dataset")

    print("\nDATASET OVERVIEW\n" + "="*60)

    total_images = 0
    total_videos = 0

    for folder in sorted(dataset_root.iterdir()):
        if not folder.is_dir():
            continue

        stats = analyze_folder(folder)

        total_images += stats["images"]
        total_videos += stats["videos"]

        label = infer_label(folder.name)

        print(f"\n=== {folder.name} ===")
        print(f"label (heuristic): {label}")
        print(f"structure: {stats['structure']}")
        print(f"images: {stats['images']}")
        print(f"videos: {stats['videos']}")
        print(f"subfolders: {stats['subfolders']}")
        print(f"folders_with_images: {stats['folders_with_images']}")
        print(f"avg_images_per_folder: {stats['avg_images_per_folder']}")

    print("\n" + "="*60)
    print(f"TOTAL images: {total_images}")
    print(f"TOTAL videos: {total_videos}")


if __name__ == "__main__":
    main()
"""
Universal SJEDD Benchmark Script
=================================
Runs SJEDD deepfake detection on any dataset with the structure:

    your_dataset/
    ├── fake/          # deepfake images OR video-frame folders
    │   ├── img001.png
    │   ├── img002.jpg
    │   └── ...        (or sub-folders of frames for video mode)
    └── real/          # pristine images OR video-frame folders
        ├── img001.png
        └── ...

Outputs: AUC, Accuracy, per-sample predictions CSV.

Usage (image mode — each file is one sample):
    python sjedd_universal_benchmark.py \
        --datapath /path/to/your_dataset \
        --resume ./pretrained/ckpt_best.pth \
        --train_dataset ffpp \
        --mode image

Usage (video mode — each sub-folder is one video, frames are averaged):
    python sjedd_universal_benchmark.py \
        --datapath /path/to/your_dataset \
        --resume ./pretrained/ckpt_best.pth \
        --train_dataset ffpp \
        --mode video \
        --n_frames 32

Arguments:
    --datapath      Path to dataset root (must contain 'fake' and 'real' sub-folders)
    --resume        Path to checkpoint (.pth). Omit to use raw CLIP weights (zero-shot).
    --train_dataset Which prompt set: 'ffpp' (default) or 'ffsc'
    --VM            CLIP backbone: ViT-B/32 (default), ViT-B/16, ViT-L/14, RN50, RN101
    --mode          'image' (default) or 'video'
    --n_frames      Frames to sample per video in video mode (default: 32)
    --batch_size    Batch size for image mode (default: 32)
    --output        Output directory for logs and CSV (default: ./output_benchmark)
    --fake_label    Folder name used as fake class (default: 'fake')
    --real_label    Folder name used as real class (default: 'real')
    --extensions    Comma-separated image extensions to scan (default: jpg,jpeg,png,bmp,webp)
"""

import os
import sys
import csv
import json
import random
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
from torch.cuda import amp
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ── sklearn and clip are required ──────────────────────────────────────────────
try:
    from sklearn.metrics import roc_auc_score, accuracy_score
except ImportError:
    sys.exit("[ERROR] scikit-learn not found. Run: pip install scikit-learn")

try:
    import clip
except ImportError:
    sys.exit("[ERROR] CLIP not found. Run: pip install git+https://github.com/openai/CLIP.git")

import torchvision.transforms as T
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC  # fallback for older torchvision


def build_clip_transform():
    """
    Robust CLIP preprocessing that always produces exactly (3, 224, 224).

    CLIP's default preprocess uses Resize(224) on the shorter side then CenterCrop(224).
    That works for square images but produces e.g. [3, 224, 398] for wide images,
    which crashes torch.stack in the DataLoader collate.

    We replace it with Resize((224, 224)) — a direct bicubic stretch to exactly 224×224 —
    which is what the original SJEDD code did in its custom_collate_fn for diffusion datasets,
    and is the correct way to handle arbitrary-aspect-ratio inputs.
    """
    return T.Compose([
        T.Resize((224, 224), interpolation=BICUBIC),
        T.ToTensor(),
        T.Normalize(
            mean=(0.48145466, 0.4578275,  0.40821073),  # CLIP's own stats
            std= (0.26862954, 0.26130258, 0.27577711),
        ),
    ])

try:
    from SO_Loss import pLoss_all_fidelity
    from SO_Graph import graph_SA_ffso, graph_SO_FFSC
except ImportError:
    sys.exit(
        "[ERROR] SO_Loss.py / SO_Graph.py not found.\n"
        "Make sure you run this script from the SJEDD repo directory."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Dataset classes
# ══════════════════════════════════════════════════════════════════════════════

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}


def _collect_images(folder: str, extensions: set) -> list:
    """Recursively find all image files in a folder (non-recursive by default)."""
    paths = []
    for entry in sorted(os.listdir(folder)):
        full = os.path.join(folder, entry)
        if os.path.isfile(full) and Path(full).suffix.lower() in extensions:
            paths.append(full)
    return paths


class UniversalImageDataset(Dataset):
    """
    Flat image dataset.
    Each image file in fake/ → label=1, real/ → label=0.
    """
    def __init__(self, datapath: str, transform, fake_label='fake', real_label='real',
                 extensions=IMAGE_EXTENSIONS):
        self.transform = transform
        self.samples = []  # list of (path, label)

        for label_name, label_int in [(real_label, 0), (fake_label, 1)]:
            folder = os.path.join(datapath, label_name)
            if not os.path.isdir(folder):
                raise FileNotFoundError(
                    f"Expected sub-folder '{label_name}' inside {datapath}.\n"
                    f"Got: {os.listdir(datapath)}"
                )
            imgs = _collect_images(folder, extensions)
            if len(imgs) == 0:
                print(f"[WARNING] No images found in {folder} with extensions {extensions}")
            for img_path in imgs:
                self.samples.append((img_path, label_int))

        print(f"[Dataset] Image mode | "
              f"real={sum(1 for _,l in self.samples if l==0)} | "
              f"fake={sum(1 for _,l in self.samples if l==1)} | "
              f"total={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, label, path


class UniversalVideoDataset(Dataset):
    """
    Video-as-folder dataset.
    Each *sub-folder* inside fake/ → label=1, real/ → label=0.
    n_frames frames are uniformly sampled from each sub-folder.
    """
    def __init__(self, datapath: str, transform, n_frames: int = 32,
                 fake_label='fake', real_label='real', extensions=IMAGE_EXTENSIONS):
        self.transform = transform
        self.n_frames = n_frames
        self.samples = []  # list of (frame_paths_list, label, folder_path)

        for label_name, label_int in [(real_label, 0), (fake_label, 1)]:
            folder = os.path.join(datapath, label_name)
            if not os.path.isdir(folder):
                raise FileNotFoundError(
                    f"Expected sub-folder '{label_name}' inside {datapath}.\n"
                    f"Got: {os.listdir(datapath)}"
                )
            for vid_name in sorted(os.listdir(folder)):
                vid_path = os.path.join(folder, vid_name)

                if not os.path.isdir(vid_path):
                    continue  # skip loose files in video mode
                frames = sorted([
                    os.path.join(vid_path, f)
                    for f in os.listdir(vid_path)
                    if Path(f).suffix.lower() in extensions
                ])
                if len(frames) == 0:
                    print(f"[WARNING] No frames found in {vid_path}, skipping.")
                    continue
                # Uniformly sample n_frames indices
                idxs = np.linspace(0, len(frames) - 1, min(n_frames, len(frames)),
                                   endpoint=True, dtype=int)
                sampled = [frames[i] for i in idxs]
                self.samples.append((sampled, label_int, vid_path))

        print(f"[Dataset] Video mode | n_frames={n_frames} | "
              f"real={sum(1 for _,l,__ in self.samples if l==0)} | "
              f"fake={sum(1 for _,l,__ in self.samples if l==1)} | "
              f"total={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames, label, vid_path = self.samples[idx]
        tensors = []
        for fpath in frames:
            img = Image.open(fpath).convert('RGB')
            tensors.append(self.transform(img))
        return torch.stack(tensors, 0), label, vid_path


# ══════════════════════════════════════════════════════════════════════════════
# Model helpers (from SJEDD_test.py)
# ══════════════════════════════════════════════════════════════════════════════

def build_text_prompts(train_dataset: str, device: str):
    l1_level = ['fake']

    if train_dataset == 'ffpp':
        l2_level = ['expression', 'identity', 'physical inconsistency']
        l3_level = ['eye', 'eyebrow', 'lip', 'mouth', 'nose', 'skin']
    elif train_dataset == 'ffsc':
        l2_level = ['age', 'expression', 'gender', 'identity', 'pose']
        l3_level = ['eye', 'eyebrow', 'lip', 'mouth', 'nose', 'skin']
    else:
        raise ValueError(f"Unknown train_dataset '{train_dataset}'. Choose 'ffpp' or 'ffsc'.")

    l1_texts = torch.cat([
        clip.tokenize(f"A photo of a {l} face") for l in l1_level
    ]).to(device).unsqueeze(0)

    l2_texts = torch.cat([
        clip.tokenize(f"A photo of a face with the global attribute of {l} altered")
        for l in l2_level
    ]).to(device).unsqueeze(0)

    l3_texts = torch.cat([
        clip.tokenize(f"A photo of a face with the local attribute of {l} altered")
        for l in l3_level
    ]).to(device).unsqueeze(0)

    return [l1_texts, l2_texts, l3_texts]


def do_batch3_relative_similarity(model, x, joint_texts):
    """Forward pass: image vs three text prompt levels, concat logits."""
    batch_size = x.size(0)
    l1_texts = joint_texts[0][0]
    l2_texts = joint_texts[1][0]
    l3_texts = joint_texts[2][0]

    logits1, _ = model.forward(x, l1_texts)
    logits2, _ = model.forward(x, l2_texts)
    logits3, _ = model.forward(x, l3_texts)

    logits = torch.cat([
        logits1.view(batch_size, -1),
        logits2.view(batch_size, -1),
        logits3.view(batch_size, -1),
    ], dim=1)
    return logits


def load_checkpoint(checkpoint_path: str, model):
    print(f"[Checkpoint] Loading from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    # Handle common checkpoint wrappers
    state_dict = ckpt
    for key in ('model', 'state_dict', 'model_state_dict'):
        if key in ckpt:
            state_dict = ckpt[key]
            break
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"[Checkpoint] Load result: missing={msg.missing_keys[:5]} "
          f"unexpected={msg.unexpected_keys[:5]}")


# ══════════════════════════════════════════════════════════════════════════════
# Inference loops
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_image_inference(model, criterion, joint_texts, data_loader, device):
    """
    Process flat image batches (like test_imgs in original code).
    Returns (all_scores, all_labels, all_paths).
    """
    model.eval()
    all_scores, all_labels, all_paths = [], [], []

    for images, labels, paths in tqdm(data_loader, desc='Images'):
        images = images.to(device, non_blocking=True)

        with amp.autocast(enabled=True):
            logits = do_batch3_relative_similarity(model, images, joint_texts)

        # criterion.infer returns (N, num_classes); [:, 0] = fake score
        print(criterion.infer(logits).shape)
        scores = criterion.infer(logits)[:, 0].detach().cpu().numpy().tolist()

        all_scores.extend(scores)
        all_labels.extend(labels.numpy().tolist())
        all_paths.extend(paths)

    return all_scores, all_labels, all_paths


@torch.no_grad()
def run_video_inference(model, criterion, joint_texts, data_loader, device):
    """
    Process video-as-folder samples (like test() in original code).
    Frames are averaged into a single video-level score.
    Returns (all_scores, all_labels, all_paths).
    """
    model.eval()
    all_scores, all_labels, all_paths = [], [], []

    for frames_batch, labels, vid_paths in tqdm(data_loader, desc='Videos'):
        # frames_batch shape: (1, n_frames, C, H, W) — batch_size is forced to 1
        frames = frames_batch.squeeze(0).to(device, non_blocking=True)  # (n_frames, C, H, W)

        with amp.autocast(enabled=True):
            logits = do_batch3_relative_similarity(model, frames, joint_texts)

        frame_scores = criterion.infer(logits)[:, 0].detach().cpu().numpy().tolist()
        video_score = float(np.mean(frame_scores))

        # frame_scores1 = criterion.infer(logits)[:, 6].detach().cpu().numpy().tolist()
        # frame_scores2 = criterion.infer(logits)[:, 7].detach().cpu().numpy().tolist()
        # video_score1 = float(np.mean(frame_scores1))
        # video_score2 = float(np.mean(frame_scores2))
        # video_score = max(video_score1, video_score2)

        all_scores.append(video_score)
        all_labels.append(int(labels[0]))
        all_paths.append(vid_paths[0])

    return all_scores, all_labels, all_paths


# ══════════════════════════════════════════════════════════════════════════════
# Metrics & output
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(scores, labels):
    auc = roc_auc_score(labels, scores) * 100
    # Threshold at 0.5 for accuracy
    preds = [1 if s >= 0.5 else 0 for s in scores]
    acc = accuracy_score(labels, preds) * 100
    return auc, acc


def save_results(output_dir: str, scores, labels, paths, auc, acc, args):
    os.makedirs(output_dir, exist_ok=True)

    # CSV of per-sample predictions
    csv_path = os.path.join(output_dir, 'predictions.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'label', 'score', 'pred'])
        for path, label, score in zip(paths, labels, scores):
            pred = 1 if score >= 0.5 else 0
            writer.writerow([path, label, f'{score:.6f}', pred])

    # Summary JSON
    summary = {
        'AUC': round(auc, 3),
        'Accuracy': round(acc, 3),
        'n_samples': len(scores),
        'n_real': int(sum(1 for l in labels if l == 0)),
        'n_fake': int(sum(1 for l in labels if l == 1)),
        'datapath': args.datapath,
        'mode': args.mode,
        'train_dataset': args.train_dataset,
        'VM': args.VM,
        'resume': args.resume,
    }
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n' + '='*50)
    print(f'  AUC      : {auc:.3f}%')
    print(f'  Accuracy : {acc:.3f}%  (threshold=0.5)')
    print(f'  Samples  : {len(scores)}  (real={summary["n_real"]}, fake={summary["n_fake"]})')
    print(f'  CSV      : {csv_path}')
    print(f'  Summary  : {summary_path}')
    print('='*50 + '\n')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description='Universal SJEDD benchmark — folder/fake + folder/real',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--datapath', required=True,
                        help='Root of dataset (must contain fake/ and real/ sub-folders)')
    parser.add_argument('--resume', default=None,
                        help='Path to SJEDD checkpoint. Omit to use raw CLIP weights.')
    parser.add_argument('--train_dataset', default='ffpp', choices=['ffpp', 'ffsc'],
                        help='Which text prompt hierarchy to use (default: ffpp)')
    parser.add_argument('--VM', default='ViT-B/32',
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'RN50', 'RN101'],
                        help='CLIP backbone (default: ViT-B/32)')
    parser.add_argument('--mode', default='image', choices=['image', 'video'],
                        help='"image": each file is a sample. '
                             '"video": each sub-folder is a video (frames averaged). '
                             '(default: image)')
    parser.add_argument('--n_frames', type=int, default=32,
                        help='Frames to uniformly sample per video (video mode only, default: 32)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for image mode (default: 32)')
    parser.add_argument('--output', default='./output_benchmark',
                        help='Directory for output CSV and summary (default: ./output_benchmark)')
    parser.add_argument('--fake_label', default='fake',
                        help='Sub-folder name for fake class (default: fake)')
    parser.add_argument('--real_label', default='real',
                        help='Sub-folder name for real class (default: real)')
    parser.add_argument('--extensions', default='jpg,jpeg,png,bmp,webp',
                        help='Comma-separated image extensions (default: jpg,jpeg,png,bmp,webp)')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print('[WARNING] No CUDA device found, running on CPU — this will be slow.')

    extensions = {'.' + e.strip().lstrip('.') for e in args.extensions.split(',')}

    # ── Load CLIP model ──────────────────────────────────────────────────────
    print(f'[Model] Loading CLIP backbone: {args.VM}')
    model, _, _, __ = clip.load(args.VM, device=device, jit=False)
    model.to(device)

    # Use robust fixed-size transform instead of CLIP's default preprocess.
    # CLIP's default Resize(224) + CenterCrop(224) only works for square images —
    # non-square inputs produce tensors like [3, 224, 398] that crash torch.stack
    # in the DataLoader collate. We use Resize((224, 224)) — a direct bicubic
    # stretch — matching what SJEDD's own custom_collate_fn did for its diffusion
    # dataset. The normalization constants are identical to CLIP's.
    preprocess = build_clip_transform()
    print('[Transform] Resize(224×224) bicubic + CLIP normalization')

    # ── Optionally load SJEDD checkpoint ────────────────────────────────────
    if args.resume:
        load_checkpoint(args.resume, model)
    else:
        print('[Checkpoint] No checkpoint provided — using raw pre-trained CLIP weights (zero-shot).')

    model.eval()

    # ── Text prompts ─────────────────────────────────────────────────────────
    print(f'[Prompts] Building text prompts for train_dataset={args.train_dataset}')
    joint_texts = build_text_prompts(args.train_dataset, device)

    # ── Loss / inference head ─────────────────────────────────────────────────
    if args.train_dataset == 'ffpp':
        criterion = pLoss_all_fidelity(hexG=graph_SA_ffso(), dataset='ffpp')
    else:
        criterion = pLoss_all_fidelity(hexG=graph_SO_FFSC(), dataset='ffsc')

    # ── Dataset & DataLoader ──────────────────────────────────────────────────
    if args.mode == 'image':
        dataset = UniversalImageDataset(
            args.datapath, preprocess,
            fake_label=args.fake_label,
            real_label=args.real_label,
            extensions=extensions,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device == 'cuda'),
            drop_last=False,
        )
        scores, labels, paths = run_image_inference(model, criterion, joint_texts, loader, device)

    else:  # video
        dataset = UniversalVideoDataset(
            args.datapath, preprocess,
            n_frames=args.n_frames,
            fake_label=args.fake_label,
            real_label=args.real_label,
            extensions=extensions,
        )
        # batch_size=1 required: videos have variable frame counts after sampling
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device == 'cuda'),
            drop_last=False,
        )
        scores, labels, paths = run_video_inference(model, criterion, joint_texts, loader, device)

    # ── Metrics ───────────────────────────────────────────────────────────────
    if len(set(labels)) < 2:
        print('[WARNING] Only one class found in labels — AUC is undefined. Showing accuracy only.')
        preds = [1 if s >= 0.5 else 0 for s in scores]
        acc = accuracy_score(labels, preds) * 100
        auc = float('nan')
    else:
        auc, acc = compute_metrics(scores, labels)

    # ── Save results ──────────────────────────────────────────────────────────
    save_results(args.output, scores, labels, paths, auc, acc, args)


if __name__ == '__main__':
    main()
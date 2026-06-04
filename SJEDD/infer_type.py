"""
FF++ L2 Attribute Evaluation
==============================
Evaluates SJEDD's l2 predictions against ground truth labels derived from
FF++ manipulation type (parsed from filename).

Dataset structure:
    root/
    ├── df/
    │   ├── 0_Deepfakes_009_027_261.png
    │   ├── 0_Face2Face_009_027_261.png
    │   ├── 0_FaceSwap_009_027_261.png
    │   ├── 0_NeuralTextures_009_027_261.png
    │   └── ...   (FaceShifter is skipped)
    └── real/
        ├── 0_real_009_261.png
        └── ...

L2 ground truth from Table I (paper):
    Deepfakes        → expression=0, identity=1, physical_inconsistency=1
    FaceSwap         → expression=0, identity=1, physical_inconsistency=1
    Face2Face        → expression=1, identity=0, physical_inconsistency=1
    NeuralTextures   → expression=1, identity=0, physical_inconsistency=1
    real             → expression=0, identity=0, physical_inconsistency=0

Usage:
    python eval_ffpp_l2.py \
        --root /path/to/ffpp_dataset \
        --resume ./pretrained/ckpt_best.pth
"""

import os
import argparse
from collections import Counter

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from sklearn.metrics import roc_auc_score

import clip
from SO_Loss import pLoss_all_fidelity
from SO_Graph import graph_SA_ffso


# ── Label definitions ──────────────────────────────────────────────────────────

L2_LABELS = ['expression', 'identity', 'physical_inconsistency']

# GT l2 vectors per manipulation: [expression, identity, physical_inconsistency]
MANIP_TO_L2 = {
    'Deepfakes':      [0, 1, 1],
    'FaceSwap':       [0, 1, 1],
    'Face2Face':      [1, 0, 1],
    'NeuralTextures': [1, 0, 1],
    'real':           [0, 0, 0],
}

SKIP = {'FaceShifter'}

SLICE_L1 = (0, 1)
SLICE_L2 = (1, 4)   # ffpp: cols 1-3


# ── Preprocessing ──────────────────────────────────────────────────────────────

def build_transform():
    return T.Compose([
        T.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(
            mean=(0.48145466, 0.4578275,  0.40821073),
            std= (0.26862954, 0.26130258, 0.27577711),
        ),
    ])


def build_text_prompts(device):
    l1_labels = ['fake']
    l2_labels = ['expression', 'identity', 'physical inconsistency']
    l3_labels = ['eye', 'eyebrow', 'lip', 'mouth', 'nose', 'skin']

    l1 = torch.cat([
        clip.tokenize(f"A photo of a {l} face") for l in l1_labels
    ]).to(device).unsqueeze(0)
    l2 = torch.cat([
        clip.tokenize(f"A photo of a face with the global attribute of {l} altered")
        for l in l2_labels
    ]).to(device).unsqueeze(0)
    l3 = torch.cat([
        clip.tokenize(f"A photo of a face with the local attribute of {l} altered")
        for l in l3_labels
    ]).to(device).unsqueeze(0)
    return [l1, l2, l3]


def do_batch3_relative_similarity(model, x, joint_texts):
    b = x.size(0)
    l1, l2, l3 = joint_texts[0][0], joint_texts[1][0], joint_texts[2][0]
    log1, _ = model.forward(x, l1)
    log2, _ = model.forward(x, l2)
    log3, _ = model.forward(x, l3)
    return torch.cat([log1.view(b, -1), log2.view(b, -1), log3.view(b, -1)], dim=1)


def load_checkpoint(path, model):
    ckpt = torch.load(path, map_location='cpu')
    state_dict = ckpt
    for k in ['model', 'state_dict', 'model_state_dict']:
        if k in ckpt:
            state_dict = ckpt[k]
            break
    model.load_state_dict(state_dict, strict=False)


# ── Dataset ────────────────────────────────────────────────────────────────────

def parse_manipulation(filename):
    name = os.path.splitext(filename)[0]
    parts = name.split('_')
    if len(parts) < 2:
        return None
    manip = parts[1]
    if manip in SKIP:
        return None
    if manip not in MANIP_TO_L2:
        print(f"[WARNING] Unknown manipulation '{manip}' in {filename}, skipping.")
        return None
    return manip


def collect_samples(root):
    samples = []
    for split in ['fake', 'real']:
        folder = os.path.join(root, split)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Expected folder: {folder}")
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            if split == 'real':
                manip = 'real'
            else:
                manip = parse_manipulation(fname)
                if manip is None:
                    continue
            l1_gt  = 0 if manip == 'real' else 1
            l2_gt  = MANIP_TO_L2[manip]
            samples.append((os.path.join(folder, fname), manip, l1_gt, l2_gt))
    return samples


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',   required=True, help='Dataset root with df/ and real/')
    parser.add_argument('--resume', required=True, help='Path to ffpp checkpoint')
    parser.add_argument('--VM', default='ViT-B/32',
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'RN50', 'RN101'])
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Device] {device}")

    # ── Model ──────────────────────────────────────────────────────────────────
    model, _, _, __ = clip.load(args.VM, device=device, jit=False)
    model.eval()
    load_checkpoint(args.resume, model)
    joint_texts = build_text_prompts(device)
    criterion   = pLoss_all_fidelity(hexG=graph_SA_ffso(), dataset='ffpp')
    transform   = build_transform()

    # ── Dataset ────────────────────────────────────────────────────────────────
    samples = collect_samples(args.root)
    manip_counts = Counter(s[1] for s in samples)
    print(f"\n[Dataset] {len(samples)} total images")
    for manip, count in sorted(manip_counts.items()):
        l2_str = ', '.join(
            f"{l}={v}" for l, v in zip(L2_LABELS, MANIP_TO_L2[manip])
        )
        print(f"  {manip:<22} {count:>5}  GT: [{l2_str}]")

    # ── Inference ──────────────────────────────────────────────────────────────
    l1_gt_all    = []
    l1_score_all = []
    l2_gt_all    = {l: [] for l in L2_LABELS}
    l2_score_all = {l: [] for l in L2_LABELS}

    # per-manipulation mean scores for breakdown table
    manip_pred = {m: {'l1': [], **{l: [] for l in L2_LABELS}}
                  for m in MANIP_TO_L2}

    for img_path, manip, l1_gt, l2_gt in tqdm(samples, desc='Inference'):
        img = Image.open(img_path).convert('RGB')
        x   = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits  = do_batch3_relative_similarity(model, x, joint_texts)
            pmargin = criterion.infer(logits)

        probs = pmargin[0].cpu().float().numpy()

        # L1
        fake_score = float(probs[SLICE_L1[0]])
        l1_gt_all.append(l1_gt)
        l1_score_all.append(fake_score)
        manip_pred[manip]['l1'].append(fake_score)

        # L2  [expression, identity, physical_inconsistency]
        l2_probs = probs[SLICE_L2[0]:SLICE_L2[1]]
        for i, label in enumerate(L2_LABELS):
            l2_gt_all[label].append(l2_gt[i])
            l2_score_all[label].append(float(l2_probs[i]))
            manip_pred[manip][label].append(float(l2_probs[i]))

    # ── Results ────────────────────────────────────────────────────────────────
    print()
    print('═' * 50)
    print('  L1  Fake/Real AUC')
    print('═' * 50)
    l1_auc = roc_auc_score(l1_gt_all, l1_score_all)
    print(f'  AUC = {l1_auc * 100:.2f}%')

    print()
    print('═' * 50)
    print('  L2  Per-attribute AUC')
    print('═' * 50)
    for label in L2_LABELS:
        gt_vec    = l2_gt_all[label]
        score_vec = l2_score_all[label]
        if len(set(gt_vec)) < 2:
            print(f'  {label:<26}  skipped (all GT values identical)')
            continue
        auc = roc_auc_score(gt_vec, score_vec)
        print(f'  {label:<26}  AUC = {auc * 100:.2f}%')

    print()
    print('═' * 50)
    print('  L2  Mean predicted score per manipulation')
    print('═' * 50)
    col_w = 12
    header = f"  {'manipulation':<22}" + ''.join(f"{l[:10]:<{col_w}}" for l in L2_LABELS) + f"{'fake_score':<{col_w}}"
    print(header)
    print('  ' + '─' * (len(header) - 2))
    for manip in ['real', 'Face2Face', 'NeuralTextures', 'Deepfakes', 'FaceSwap']:
        entry = manip_pred.get(manip)
        if not entry or not entry['l1']:
            continue
        row = f"  {manip:<22}"
        for label in L2_LABELS:
            row += f"{np.mean(entry[label]):<{col_w}.4f}"
        row += f"{np.mean(entry['l1']):<{col_w}.4f}"
        print(row)
    print()


if __name__ == '__main__':
    main()
"""
SJEDD Single Image Inference + Interpretation
==============================================
Runs inference on one image and prints:
  - Fake/Real probability
  - Top global manipulation (l2) with probability
  - Top local manipulation (l3) with probability
  - Full probability breakdown for all levels

Usage:
    python sjedd_infer_single.py \
        --image /path/to/face.png \
        --resume ./pretrained/ckpt_best.pth \
        --train_dataset ffpp

    # ffsc checkpoint:
    python sjedd_infer_single.py \
        --image /path/to/face.png \
        --resume ./pretrained/ckpt_best_FFSC.pth \
        --train_dataset ffsc
"""

import os
import sys
import argparse
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

try:
    import clip
except ImportError:
    sys.exit("[ERROR] CLIP not found. Run: pip install git+https://github.com/openai/CLIP.git")

try:
    from SO_Loss import pLoss_all_fidelity
    from SO_Graph import graph_SA_ffso, graph_SO_FFSC
except ImportError:
    sys.exit("[ERROR] SO_Loss.py / SO_Graph.py not found. Run from the SJEDD repo directory.")


# ── Label definitions ──────────────────────────────────────────────────────────

LABELS = {
    'ffpp': {
        'l1': ['fake'],
        'l2': ['expression', 'identity', 'physical inconsistency'],
        'l3': ['eye', 'eyebrow', 'lip', 'mouth', 'nose', 'skin'],
    },
    'ffsc': {
        'l1': ['fake'],
        'l2': ['age', 'expression', 'gender', 'identity', 'pose'],
        'l3': ['eye', 'eyebrow', 'lip', 'mouth', 'nose', 'skin'],
    },
}

# pMargin column layout (from transfer_logtis in SO_Loss.py):
#   ffsc:  col 0       → l1 (fake prob)
#          cols 1-5    → l2 (5 global attrs)
#          cols 6-11   → l3 (6 local attrs)
#
#   ffpp:  col 0       → l1 (fake prob)
#          cols 1-3    → l2 (3 global attrs)
#          cols 4-9    → l3 (6 local attrs)

SLICE = {
    'ffsc': {'l1': (0, 1),  'l2': (1, 6),  'l3': (6, 12)},
    'ffpp': {'l1': (0, 1),  'l2': (1, 4),  'l3': (4, 10)},
}


# ── Preprocessing ──────────────────────────────────────────────────────────────

def build_transform():
    """Fixed 224×224 bicubic resize + CLIP normalization."""
    return T.Compose([
        T.Resize((224, 224), interpolation=BICUBIC),
        T.ToTensor(),
        T.Normalize(
            mean=(0.48145466, 0.4578275,  0.40821073),
            std= (0.26862954, 0.26130258, 0.27577711),
        ),
    ])


# ── Model helpers ──────────────────────────────────────────────────────────────

def build_text_prompts(train_dataset, device):
    labels = LABELS[train_dataset]
    l1_texts = torch.cat([
        clip.tokenize(f"A photo of a {l} face") for l in labels['l1']
    ]).to(device).unsqueeze(0)
    l2_texts = torch.cat([
        clip.tokenize(f"A photo of a face with the global attribute of {l} altered")
        for l in labels['l2']
    ]).to(device).unsqueeze(0)
    l3_texts = torch.cat([
        clip.tokenize(f"A photo of a face with the local attribute of {l} altered")
        for l in labels['l3']
    ]).to(device).unsqueeze(0)
    return [l1_texts, l2_texts, l3_texts]


def do_batch3_relative_similarity(model, x, joint_texts):
    b = x.size(0)
    l1, l2, l3 = joint_texts[0][0], joint_texts[1][0], joint_texts[2][0]
    log1, _ = model.forward(x, l1)
    log2, _ = model.forward(x, l2)
    log3, _ = model.forward(x, l3)
    return torch.cat([log1.view(b, -1), log2.view(b, -1), log3.view(b, -1)], dim=1)


def load_checkpoint(path, model):
    print(f"[Checkpoint] Loading {path}")
    ckpt = torch.load(path, map_location='cpu')
    state_dict = ckpt
    for key in ('model', 'state_dict', 'model_state_dict'):
        if key in ckpt:
            state_dict = ckpt[key]
            break
    msg = model.load_state_dict(state_dict, strict=False)
    if msg.missing_keys:
        print(f"[Checkpoint] Missing keys (first 5): {msg.missing_keys[:5]}")


# ── Pretty printer ─────────────────────────────────────────────────────────────

BAR_WIDTH = 30

def prob_bar(p):
    filled = int(round(p * BAR_WIDTH))
    return '█' * filled + '░' * (BAR_WIDTH - filled)

def print_results(pmargin, train_dataset):
    probs = pmargin[0].cpu().float().numpy()  # shape (n,)
    labels = LABELS[train_dataset]
    sl = SLICE[train_dataset]

    fake_prob = float(probs[sl['l1'][0]])
    real_prob = 1.0 - fake_prob

    verdict = 'FAKE' if fake_prob >= 0.5 else 'REAL'
    confidence = max(fake_prob, real_prob) * 100

    # ── Header ────────────────────────────────────────────────────────────────
    print()
    print('╔══════════════════════════════════════════════════╗')
    print(f'║  Verdict : {verdict:4s}  ({confidence:.1f}% confidence)'.ljust(51) + '║')
    print('╚══════════════════════════════════════════════════╝')

    # ── L1: fake / real ────────────────────────────────────────────────────────
    print()
    print('─── L1 : Fake / Real ───────────────────────────────')
    print(f'  {"fake":<26}  {fake_prob:.4f}  {prob_bar(fake_prob)}')
    print(f'  {"real":<26}  {real_prob:.4f}  {prob_bar(real_prob)}')

    # ── L2: global attributes ─────────────────────────────────────────────────
    l2_probs = probs[sl['l2'][0]:sl['l2'][1]]
    l2_labels = labels['l2']
    l2_top_idx = int(np.argmax(l2_probs))

    print()
    print('─── L2 : Global manipulation type ─────────────────')
    for i, (name, p) in enumerate(zip(l2_labels, l2_probs)):
        marker = ' ◄ top' if i == l2_top_idx else ''
        print(f'  {name:<26}  {p:.4f}  {prob_bar(p)}{marker}')

    # ── L3: local attributes ──────────────────────────────────────────────────
    l3_probs = probs[sl['l3'][0]:sl['l3'][1]]
    l3_labels = labels['l3']
    l3_top_idx = int(np.argmax(l3_probs))

    print()
    print('─── L3 : Local region affected ─────────────────────')
    for i, (name, p) in enumerate(zip(l3_labels, l3_probs)):
        marker = ' ◄ top' if i == l3_top_idx else ''
        print(f'  {name:<26}  {p:.4f}  {prob_bar(p)}{marker}')

    # ── Summary line ──────────────────────────────────────────────────────────
    print()
    print('─── Summary ─────────────────────────────────────────')
    print(f'  Verdict  : {verdict} ({fake_prob*100:.1f}% fake)')
    print(f'  Top L2   : {l2_labels[l2_top_idx]} ({l2_probs[l2_top_idx]*100:.1f}%)')
    print(f'  Top L3   : {l3_labels[l3_top_idx]} ({l3_probs[l3_top_idx]*100:.1f}%)')
    print()

    return {
        'verdict': verdict,
        'fake_prob': fake_prob,
        'real_prob': real_prob,
        'l2': dict(zip(l2_labels, l2_probs.tolist())),
        'l3': dict(zip(l3_labels, l3_probs.tolist())),
        'top_l2': l2_labels[l2_top_idx],
        'top_l3': l3_labels[l3_top_idx],
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='SJEDD single image inference')
    parser.add_argument('--image', required=True, help='Path to input face image')
    parser.add_argument('--resume', default=None, help='Path to SJEDD checkpoint')
    parser.add_argument('--train_dataset', default='ffpp', choices=['ffpp', 'ffsc'])
    parser.add_argument('--VM', default='ViT-B/32',
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'RN50', 'RN101'])
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.image):
        sys.exit(f"[ERROR] Image not found: {args.image}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Device] {device}")

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"[Model] Loading CLIP {args.VM}")
    model, _, _, __ = clip.load(args.VM, device=device, jit=False)
    model.to(device).eval()

    if args.resume:
        load_checkpoint(args.resume, model)
    else:
        print("[Checkpoint] No checkpoint — using raw CLIP weights (zero-shot)")

    # ── Text prompts & criterion ───────────────────────────────────────────────
    joint_texts = build_text_prompts(args.train_dataset, device)

    if args.train_dataset == 'ffpp':
        criterion = pLoss_all_fidelity(hexG=graph_SA_ffso(), dataset='ffpp')
    else:
        criterion = pLoss_all_fidelity(hexG=graph_SO_FFSC(), dataset='ffsc')

    # ── Load & preprocess image ────────────────────────────────────────────────
    transform = build_transform()
    img = Image.open(args.image).convert('RGB')
    print(f"[Image] {args.image}  (original size: {img.size[0]}×{img.size[1]})")

    x = transform(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    # ── Inference ──────────────────────────────────────────────────────────────
    with torch.no_grad():
        from torch.cuda import amp
        with amp.autocast(enabled=(device == 'cuda')):
            logits = do_batch3_relative_similarity(model, x, joint_texts)
        pmargin = criterion.infer(logits)   # (1, n_cols)

    print(f"[Debug] logits shape: {logits.shape}  |  pmargin shape: {pmargin.shape}")

    # ── Print results ──────────────────────────────────────────────────────────
    result = print_results(pmargin, args.train_dataset)
    return result


if __name__ == '__main__':
    main()
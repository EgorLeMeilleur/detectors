"""
SJEDD Folder Inference
======================

Runs inference on all images inside a folder and outputs
a list of fake probabilities (one per image).

Usage:
    python sjedd_infer_folder.py \
        --folder /path/to/images \
        --resume ./pretrained/ckpt_best.pth \
        --train_dataset ffpp \
        --output fake_probs.txt
"""

import os
import sys
import argparse
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

try:
    import clip
except ImportError:
    sys.exit("[ERROR] CLIP not found.")

try:
    from SO_Loss import pLoss_all_fidelity
    from SO_Graph import graph_SA_ffso, graph_SO_FFSC
except ImportError:
    sys.exit("[ERROR] Run from SJEDD repo directory.")


SLICE = {
    'ffsc': {'l1': (0, 1)},
    'ffpp': {'l1': (0, 1)},
}


# ─────────────────────────────────────────────────────────

def build_transform():
    return T.Compose([
        T.Resize((224, 224), interpolation=BICUBIC),
        T.ToTensor(),
        T.Normalize(
            mean=(0.48145466, 0.4578275,  0.40821073),
            std= (0.26862954, 0.26130258, 0.27577711),
        ),
    ])


def build_text_prompts(train_dataset, device):
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

    labels = LABELS[train_dataset]

    l1 = torch.cat([
        clip.tokenize(f"A photo of a {l} face")
        for l in labels['l1']
    ]).to(device).unsqueeze(0)

    l2 = torch.cat([
        clip.tokenize(f"A photo of a face with the global attribute of {l} altered")
        for l in labels['l2']
    ]).to(device).unsqueeze(0)

    l3 = torch.cat([
        clip.tokenize(f"A photo of a face with the local attribute of {l} altered")
        for l in labels['l3']
    ]).to(device).unsqueeze(0)

    return [l1, l2, l3]


def do_batch3_relative_similarity(model, x, joint_texts):
    b = x.size(0)
    l1, l2, l3 = joint_texts[0][0], joint_texts[1][0], joint_texts[2][0]

    log1, _ = model.forward(x, l1)
    log2, _ = model.forward(x, l2)
    log3, _ = model.forward(x, l3)

    return torch.cat([
        log1.view(b, -1),
        log2.view(b, -1),
        log3.view(b, -1)
    ], dim=1)


def load_checkpoint(path, model):
    ckpt = torch.load(path, map_location='cpu')
    state_dict = ckpt
    for key in ('model', 'state_dict', 'model_state_dict'):
        if key in ckpt:
            state_dict = ckpt[key]
            break
    model.load_state_dict(state_dict, strict=False)


# ─────────────────────────────────────────────────────────
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True, help='Folder with images')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--train_dataset', default='ffpp', choices=['ffpp', 'ffsc'])
    parser.add_argument('--VM', default='ViT-B/32',
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'RN50', 'RN101'])
    parser.add_argument('--output', default=None, help='Optional output txt file')
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.folder):
        sys.exit("[ERROR] Folder not found")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load CLIP
    model, _, _, __ = clip.load(args.VM, device=device, jit=False)
    model.to(device).eval()

    if args.resume:
        load_checkpoint(args.resume, model)

    # Text prompts + criterion
    joint_texts = build_text_prompts(args.train_dataset, device)

    if args.train_dataset == 'ffpp':
        criterion = pLoss_all_fidelity(hexG=graph_SA_ffso(), dataset='ffpp')
    else:
        criterion = pLoss_all_fidelity(hexG=graph_SO_FFSC(), dataset='ffsc')

    transform = build_transform()

    image_paths = sorted([
        os.path.join(args.folder, f)
        for f in os.listdir(args.folder)
    ])

    fake_probs = []

    with torch.no_grad():
        for path in tqdm(image_paths):
            fake_prob_ = []
            for img_path in os.listdir(path):
                img = Image.open(os.path.join(path, img_path)).convert('RGB')
                x = transform(img).unsqueeze(0).to(device)

                logits = do_batch3_relative_similarity(model, x, joint_texts)
                pmargin = criterion.infer(logits)

                fake_prob = float(pmargin[0, SLICE[args.train_dataset]['l1'][0]].cpu())
                fake_prob_.append(fake_prob)
            fake_probs.append(np.mean(fake_prob_))

    # Optional save
    if args.output:
        with open(args.output, 'w') as f:
            for p in fake_probs:
                f.write(f"{p}\n")


if __name__ == "__main__":
    main()
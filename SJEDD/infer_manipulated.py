import os
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
import cv2

import clip

from SO_Loss import pLoss_all_fidelity
from SO_Graph import graph_SA_ffso, graph_SO_FFSC
from preprocessing.face_utils2 import FaceDetector, norm_crop


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

SLICE = {
    'ffsc': {'l1': (0, 1), 'l2': (1, 6), 'l3': (6, 12)},
    'ffpp': {'l1': (0, 1), 'l2': (1, 4), 'l3': (4, 10)},
}


# -------------------------------------------------------
# Face detector
# -------------------------------------------------------

face_detector = FaceDetector(device='cuda:0')
face_detector.load_checkpoint("preprocessing/RetinaFace-Resnet50-fixed.pth")


def preprocess_face(image):

    img = np.array(image)

    boxes, landms = face_detector.detect(img)

    if boxes.shape[0] == 0:
        img = cv2.resize(img, (317, 317))
        return Image.fromarray(img)

    areas = (boxes[:,3]-boxes[:,1])*(boxes[:,2]-boxes[:,0])
    idx = areas.argmax()

    landm = landms[idx]
    landmarks = landm.detach().cpu().numpy().reshape(5,2).astype(int)

    aligned = norm_crop(img, landmarks, outsize=(317,317))

    return Image.fromarray(aligned)


# -------------------------------------------------------
# Transform (MATCHES MAIN SCRIPT)
# -------------------------------------------------------

from torchvision.transforms import InterpolationMode

def build_transform():
    return T.Compose([
        T.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(
            mean=(0.48145466, 0.4578275,  0.40821073),
            std= (0.26862954, 0.26130258, 0.27577711)
        )
    ])


# -------------------------------------------------------
# Text prompts (MATCHES MAIN SCRIPT)
# -------------------------------------------------------

def build_text_prompts(train_dataset, device):

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

    return [l1,l2,l3]


# -------------------------------------------------------
# CLIP similarity
# -------------------------------------------------------

def do_batch3_relative_similarity(model, x, joint_texts):

    b = x.size(0)

    l1,l2,l3 = joint_texts[0][0], joint_texts[1][0], joint_texts[2][0]

    log1,_ = model.forward(x,l1)
    log2,_ = model.forward(x,l2)
    log3,_ = model.forward(x,l3)

    return torch.cat([
        log1.view(b,-1),
        log2.view(b,-1),
        log3.view(b,-1)
    ], dim=1)


# -------------------------------------------------------
# Checkpoint
# -------------------------------------------------------

def load_checkpoint(path, model):

    ckpt = torch.load(path,map_location="cpu")

    state_dict = ckpt
    for k in ['model','state_dict','model_state_dict']:
        if k in ckpt:
            state_dict = ckpt[k]

    model.load_state_dict(state_dict,strict=False)


# -------------------------------------------------------
# Label parser
# -------------------------------------------------------

def parse_gt_label(filename):

    name = os.path.splitext(filename)[0]

    if name.endswith("lip_mouth"):
        return "lip_mouth"
    elif name.endswith("eye"):
        return "eye"
    elif name.endswith("nose"):
        return "nose"

    raise ValueError(filename)


# -------------------------------------------------------
# Main
# -------------------------------------------------------
from sklearn.metrics import roc_auc_score
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir",required=True)
    parser.add_argument("--resume",required=True)
    parser.add_argument('--train_dataset', default='ffpp',
                        choices=['ffpp','ffsc'])

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:",device)

    model,_ ,_,__ = clip.load("ViT-B/32",device=device,jit=False)
    model.eval()

    load_checkpoint(args.resume,model)

    joint_texts = build_text_prompts(args.train_dataset,device)

    if args.train_dataset == "ffpp":
        criterion = pLoss_all_fidelity(
            hexG=graph_SA_ffso(),
            dataset='ffpp'
        )
    else:
        criterion = pLoss_all_fidelity(
            hexG=graph_SO_FFSC(),
            dataset='ffsc'
        )

    transform = build_transform()

    images = [f for f in os.listdir(args.image_dir)
              if f.endswith((".jpg",".png",".jpeg"))]

    y_true = []
    y_pred = []

    classes = ['eye', 'lip_mouth', 'nose']
    results = {c: {'y_true': [], 'y_score': []} for c in classes}

    for img_name in tqdm(images):

        gt = parse_gt_label(img_name)

        img_path = os.path.join(args.image_dir,img_name)

        img = Image.open(img_path).convert("RGB")

        # -------- FACE ALIGNMENT --------
        #img = preprocess_face(img)

        # -------- TRANSFORM --------
        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():

            logits = do_batch3_relative_similarity(
                model,x,joint_texts
            )

            pmargin = criterion.infer(logits)

        probs = pmargin[0].cpu().numpy()
        sl = SLICE[args.train_dataset]
        l3_probs = probs[sl['l3'][0]:sl['l3'][1]]
        l3_labels = LABELS[args.train_dataset]['l3']
        prob_dict = dict(zip(l3_labels, l3_probs))
 
        eye_score      = prob_dict['eye']
        lip_mouth_score = max(prob_dict['lip'], prob_dict['mouth'])
        nose_score     = prob_dict['nose']
 
        scores = {
            'eye':       eye_score,
            'lip_mouth': lip_mouth_score,
            'nose':      nose_score,
        }
 
        # For each class: binary label (is this the gt?) and the model's score for it
        for c in classes:
            results[c]['y_true'].append(1 if gt == c else 0)
            results[c]['y_score'].append(scores[c])
 
    # ── Per-class AUC ──────────────────────────────────────────────────────────
    print("\n── Per-class ROC AUC ─────────────────────────")
    aucs = []
    for c in classes:
        y_true  = results[c]['y_true']
        y_score = results[c]['y_score']
        if len(set(y_true)) < 2:
            print(f"  {c:<12}  skipped (only one class in labels)")
            continue
        auc = roc_auc_score(y_true, y_score)
        aucs.append(auc)
        print(f"  {c:<12}  AUC = {auc:.4f}")
 
    if aucs:
        print(f"\n  Macro avg AUC = {np.mean(aucs):.4f}")
 
    # ── Multiclass AUC (one-vs-rest, macro) ────────────────────────────────────
    all_y_true  = [parse_gt_label(f) for f in images]
    label_to_int = {c: i for i, c in enumerate(classes)}
    y_true_int  = [label_to_int[l] for l in all_y_true]
    y_score_mat = np.column_stack([results[c]['y_score'] for c in classes])
 
    try:
        auc_ovr = roc_auc_score(y_true_int, y_score_mat, multi_class='ovr', average='macro')
        print(f"  Multiclass OvR macro AUC = {auc_ovr:.4f}")
    except Exception as e:
        print(f"  Multiclass AUC failed: {e}")
 
    print()
 
 
if __name__ == "__main__":
    main()
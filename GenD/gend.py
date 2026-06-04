import requests
import torch
from PIL import Image

from src.hf.modeling_gend import GenD

# Other models can be found in https://huggingface.co/collections/yermandy/gend
# - yermandy/GenD_CLIP_L_14
# - yermandy/GenD_PE_L
# - yermandy/GenD_DINOv3_L
model = GenD.from_pretrained("yermandy/GenD_CLIP_L_14")

urls = [
    "https://github.com/yermandy/deepfake-detection/blob/main/datasets/FF/DF/000_003/000.png?raw=true",
    "https://github.com/yermandy/deepfake-detection/blob/main/datasets/FF/real/000/000.png?raw=true",
]
images = [Image.open(requests.get(url, stream=True).raw) for url in urls]
tensors = torch.stack([model.feature_extractor.preprocess(img) for img in images])
logits = model(tensors)
probs = logits.softmax(dim=-1)

print(probs)
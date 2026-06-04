import pdb

import cv2
import numpy as np

import torch
from skimage import transform as trans

from preprocessing.external.Pytorch_Retinaface.data import cfg_re50 # retinaface/config
from preprocessing.external.Pytorch_Retinaface.layers.functions.prior_box import PriorBox # retinaface
from preprocessing.external.Pytorch_Retinaface.utils.box_utils import decode, decode_landm # retinaface
from preprocessing.external.Pytorch_Retinaface.models.retinaface import RetinaFace # retinaface

def norm_crop(img, landmark=None, outsize=None, scale=1.3, mask=None):
    """ align and crop the face according to the given bbox and landmarks
        landmark: 5 key points
    """

    M = None

    target_size = [112, 112]

    dst = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32)

    if target_size[1] == 112:
        dst[:, 0] += 8.0

    dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
    dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]

    target_size = outsize

    margin_rate = scale - 1
    x_margin = target_size[0] * margin_rate / 2.
    y_margin = target_size[1] * margin_rate / 2.

    # move
    dst[:, 0] += x_margin
    dst[:, 1] += y_margin

    # resize
    dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
    dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

    src = landmark.astype(np.float32)

    # use skimage tranformation
    tform = trans.SimilarityTransform()
    tform.estimate(src, dst)
    M = tform.params[0:2, :]

    # M: use opencv
    # M = cv2.getAffineTransform(src[[0,1,2],:],dst[[0,1,2],:])

    img = cv2.warpAffine(img, M, (target_size[1], target_size[0]))

    if outsize is not None:
        img = cv2.resize(img, (outsize[1], outsize[0]))

    if mask is not None:
        mask = cv2.warpAffine(mask, M, (target_size[1], target_size[0]))
        mask = cv2.resize(mask, (outsize[1], outsize[0]))
        return img, mask
    else:
        return img

class FaceDetector:
    def __init__(self, device="cuda", confidence_threshold=0.8):
        self.device = device
        self.confidence_threshold = confidence_threshold

        self.cfg = cfg = cfg_re50
        self.variance = cfg["variance"]
        cfg["pretrain"] = False

        self.net = RetinaFace(cfg=cfg, phase="test").to(device).eval()
        self.decode_param_cache = {}

    def load_checkpoint(self, path):
        self.net.load_state_dict(torch.load(path))

    def decode_params(self, height, width):
        cache_key = (height, width)

        try:
            return self.decode_param_cache[cache_key]
        except KeyError:
            priorbox = PriorBox(self.cfg, image_size=(height, width))
            priors = priorbox.forward()

            prior_data = priors.data
            scale = torch.Tensor([width, height] * 2)
            scale1 = torch.Tensor([width, height] * 5)

            result = (prior_data, scale, scale1)
            self.decode_param_cache[cache_key] = result

            return result

    def detect(self, img):
        device = self.device

        prior_data, scale, scale1 = self.decode_params(*img.shape[:2])

        # REF: test_fddb.py
        img = np.float32(img)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device, dtype=torch.float32)

        loc, conf, landms = self.net(img)

        loc = loc.cpu()
        conf = conf.cpu()
        landms = landms.cpu()

        # Decode results
        boxes = decode(loc.squeeze(0), prior_data, self.variance)
        boxes = boxes * scale
        scores = conf.squeeze(0)[:, 1]

        landms = decode_landm(landms.squeeze(0), prior_data, self.variance)
        landms = landms * scale1

        inds = scores > self.confidence_threshold
        boxes = boxes[inds]
        landms = landms[inds]

        return boxes, landms

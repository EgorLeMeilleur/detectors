import torch, random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2, math


def set_dataset_singleGPU(txt_path, preprocess, config, args, phase='train', aug=False, mask=False):
    if not aug:
        dataset = clip_deepfake_data(txt_path, preprocess, mask=mask)
    else:
        dataset = clip_deepfake_data_aug(txt_path, preprocess, aug_probs=args.aug_probs, mask=mask)

    print(f"successfully build {phase} dataset")

    if phase == 'train':
        data_loader_train = torch.utils.data.DataLoader(
            dataset, shuffle=True,
            batch_size=config.DATA.BATCH_SIZE,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=True,
        )
        return dataset, data_loader_train

    elif phase == 'val':
        data_loader_val = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=False
        )
        return dataset, data_loader_val


class clip_deepfake_data(Dataset):
    def __init__(self, txt_path, preprocess, mask=False):
        super().__init__()
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            if len(words) == 0:
                continue
            try:
                imgs.append((words[0], words[1], words[2]))
            except:
                imgs.append((words[0], words[1]))

        self.imgs = imgs
        self.preprocess = preprocess
        self.mask = mask

    def __getitem__(self, index):
        try:
            fn, flag, label = self.imgs[index]
        except:
            fn, label = self.imgs[index]

        img = Image.open(fn).convert('RGB')

        if self.preprocess is not None:
            img = self.preprocess(img)

        if self.mask:
            mask = self.get_label_mask(label)
            return img, label, mask

        return img, label

    def __len__(self):
        return len(self.imgs)

    def get_label_mask(self, label):
        label = [int(char) for char in label]
        label_out = np.array(label)
        mask = np.ones_like(label_out)
        mask[label_out == 2] = 0

        return mask


class clip_deepfake_data_aug(Dataset):
    def __init__(self, txt_path, preprocess, aug_probs=0.3, mask=False):
        super().__init__()
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            if len(words) == 0:
                continue
            try:
                imgs.append((words[0], words[1], words[2]))
            except:
                imgs.append((words[0], words[1]))

        self.imgs = imgs
        self.preprocess = preprocess
        self.aug_probs = aug_probs
        self.mask = mask

    def __getitem__(self, index):
        try:
            fn, flag, label = self.imgs[index]
        except:
            fn, label = self.imgs[index]

        img = Image.open(fn).convert('RGB')

        if random.random() < self.aug_probs:
            img = self.perturbation_jpeg(img)

        if self.preprocess is not None:
            img = self.preprocess(img)

        if self.mask:
            mask = self.get_label_mask(label)
            return img, label, mask

        return img, label

    def __len__(self):
        return len(self.imgs)

    def get_label_mask(self, label):
        label = [int(char) for char in label]
        label_out = np.array(label)
        mask = np.ones_like(label_out)
        mask[label_out == 2] = 0

        return mask

    def get_perturb(self, type, level):
        def get_distortion_function(type):
            func_dict = dict()  # a dict of function
            func_dict['BW'] = block_wise
            func_dict['GNC'] = gaussian_noise_color
            func_dict['GB'] = gaussian_blur
            func_dict['JPEG'] = pixelation

            return func_dict[type]

        def get_distortion_parameter(type, level):
            param_dict = dict()  # a dict of list
            param_dict['BW'] = [16, 32]  # larger, worse
            param_dict['GNC'] = [0.001, 0.002]  # larger, worse
            param_dict['GB'] = [7, 9]  # larger, worse
            param_dict['JPEG'] = [2, 3]  # larger, worse
            # level starts from 1, list starts from 0
            return param_dict[type][level - 1]

        level = int(level)
        dist_function = get_distortion_function(type)
        dist_param = get_distortion_parameter(type, level)
        return dist_function, dist_param

    def perturbation_jpeg(self, im):
        distortion = ['BW', 'GNC', 'GB', 'JPEG', 'RealJPEG']
        type = random.choice(distortion)
        if random.random() < 0.5:
            level = 1
        else:
            level = 2

        if type != 'RealJPEG':
            im = np.asarray(im)
            im = np.copy(im)
            im = np.flip(im, 2)

            dist_function, dist_param = self.get_perturb(type, level)
            im = dist_function(im, dist_param)
            im = Image.fromarray(np.flip(im, 2))

            return im
        else: # REAL JPEG
            im = cv2.cvtColor(np.array(im), cv2.COLOR_RGBA2BGRA)  # PIL转cv2
            quality = random.choice([75, 80, 90])
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            face_img_encode = cv2.imencode('.jpg', im, encode_param)[1]
            im = cv2.imdecode(face_img_encode, cv2.IMREAD_COLOR)
            im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            return im

    def perturbation(self, im):
        distortion = ['BW', 'GNC', 'GB', 'JPEG']
        # distortion = ['BW', 'GNC', 'GNC', 'GB', 'JPEG', 'GB', 'JPEG']
        type = random.choice(distortion)
        if random.random() < 0.5:
            level = 1
        else:
            level = 2

        im = np.asarray(im)
        im = np.copy(im)
        im = np.flip(im, 2)

        dist_function, dist_param = self.get_perturb(type, level)
        im = dist_function(im, dist_param)

        im = Image.fromarray(np.flip(im, 2))

        return im


# ==========================================================================================



def stack_label_perturb(targets):
    label_ls = []
    for str_label in targets:
        label = list(map(int, str_label))
        label = torch.tensor(label, dtype=torch.float)
        label_ls.append(label)
    label_cat = torch.stack(label_ls)
    return label_cat



label_map_soff = {
    '1000000000': [0, 0,0,0, 0,0,0,0,0,0], # real
    '0100110000': [1, 0,0,1, 1,0,0,0,0,0], # phys_eyes
    '0110001100': [1, 1,0,1, 0,0,1,1,0,0], # expr_mouth -- nt, f2f
    '0100101100': [1, 0,0,1, 0,0,1,1,0,0], # phy_mouth
    '0100100010': [1, 0,0,1, 0,0,0,0,1,0], # phy_nose
    '0101011111': [1, 0,1,1, 1,1,1,1,1,1], # id -- df, fs
}

label_map_ffsc = {
    '110000100001': [0,1,1,0,0,0,0,1,0,0,0,0,1], # age
    '101000001100': [0,1,0,1,0,0,0,0,0,1,1,0,0], # expr-smile
    '101000101111': [0,1,0,1,0,0,0,1,0,1,1,1,1], # expr-surprise
    '120110111101': [0,1,0,0,1,1,0,1,1,1,1,0,1], # gender
    '120010111111': [0,1,0,0,0,1,0,1,1,1,1,1,1], # id
    '100001111111': [0,1,0,0,0,0,1,1,1,1,1,1,1], # pose
    '000000000000': [1,0,0,0,0,0,0,0,0,0,0,0,0] # real
}
label_map_ffsc_so = {
    '110000100001': [1,1,0,0,0,0,1,0,0,0,0,1], # age
    '101000001100': [1,0,1,0,0,0,0,0,1,1,0,0], # expr-smile
    '101000101111': [1,0,1,0,0,0,1,0,1,1,1,1], # expr-surprise
    '120110111101': [1,0,0,1,1,0,1,1,1,1,0,1], # gender
    '120010111111': [1,0,0,0,1,0,1,1,1,1,1,1], # id
    '100001111111': [1,0,0,0,0,1,1,1,1,1,1,1], # pose
    '000000000000': [0,0,0,0,0,0,0,0,0,0,0,0] # real
}

def stack_label(targets, dataset=''):
    if dataset in ['ffsc']:
        label_map = label_map_ffsc
    elif dataset in ['ffsc-so']:
        label_map = label_map_ffsc_so
    else:
        label_map = label_map_soff
    label_ls = []
    for str_label in targets:
        label = label_map[str_label]
        label = torch.tensor(label)
        label_ls.append(label)
    label_cat = torch.stack(label_ls)
    return label_cat





# utils functions of distortions
def bgr2ycbcr(img_bgr):
    img_bgr = img_bgr.astype(np.float32)
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCR_CB)
    img_ycbcr = img_ycrcb[:, :, (0, 2, 1)].astype(np.float32)
    # to [16/255, 235/255]
    img_ycbcr[:, :, 0] = (img_ycbcr[:, :, 0] * (235 - 16) + 16) / 255.0
    # to [16/255, 240/255]
    img_ycbcr[:, :, 1:] = (img_ycbcr[:, :, 1:] * (240 - 16) + 16) / 255.0

    return img_ycbcr

def ycbcr2bgr(img_ycbcr):
    img_ycbcr = img_ycbcr.astype(np.float32)
    # to [0, 1]
    img_ycbcr[:, :, 0] = (img_ycbcr[:, :, 0] * 255.0 - 16) / (235 - 16)
    # to [0, 1]
    img_ycbcr[:, :, 1:] = (img_ycbcr[:, :, 1:] * 255.0 - 16) / (240 - 16)
    img_ycrcb = img_ycbcr[:, :, (0, 2, 1)].astype(np.float32)
    img_bgr = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCR_CB2BGR)

    return img_bgr

# distortion functions
def block_wise(img, param):
    width = 8
    block = np.ones((width, width, 3)).astype(int) * 128
    param = min(img.shape[0], img.shape[1]) // 256 * param
    for i in range(param):
        r_w = random.randint(0, img.shape[1] - 1 - width)
        r_h = random.randint(0, img.shape[0] - 1 - width)
        # pdb.set_trace()
        img[r_h:r_h + width, r_w:r_w + width, :] = block

    return img

def gaussian_noise_color(img, param):
    ycbcr = bgr2ycbcr(img) / 255
    size_a = ycbcr.shape
    b = (ycbcr + math.sqrt(param) *
         np.random.randn(size_a[0], size_a[1], size_a[2])) * 255
    b = ycbcr2bgr(b)
    img = np.clip(b, 0, 255).astype(np.uint8)

    return img

def gaussian_blur(img, param):
    img = cv2.GaussianBlur(img, (param, param), param * 1.0 / 6)

    return img

def pixelation(img, param):
    h, w, _ = img.shape
    s_h = h // param
    s_w = w // param
    img = cv2.resize(img, (s_w, s_h))
    img = cv2.resize(img, (w, h))

    return img
# --------------------------------------------------------------------------------------------------------------------

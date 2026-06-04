import json
import torch, random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os
import pandas as pd

class MyDataset_FFSC(torch.utils.data.Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None, output_addr=False):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            if len(words) == 0:
                continue

            label = int(words[1][0])
            imgs.append((words[0], label))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.output_addr = output_addr

        print(f"successfully build FFSC test dataset...")

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.output_addr:
            return img, label, fn

        return img, label

    def __len__(self):
        return len(self.imgs)

class MyDataset_Diffusion(torch.utils.data.Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None, output_addr=False):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            if len(words) == 0:
                continue

            label = int(words[1])
            imgs.append((words[0], label))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.output_addr = output_addr

        print(f"successfully build diffusion test dataset...")

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        # img = self.perturbation(img, self.perturb)

        img = self.transform(img)

        if self.output_addr:
            return img, label, fn

        return img, label

    def __len__(self):
        return len(self.imgs)

# ------------------------------------------------------------------------------------------------------------------- #
# for Celeb-DF dataset
class BuildCelebDFdataset_online(Dataset):
    def __init__(self, preprocess, datapath = '/data/CDF/faces/', n_frames=32):
        super().__init__()

        self.n_frames = n_frames

        self.datapath = datapath
        self.vid_file_path = []
        self.label_list = []

        d_type = ['RealCDF', 'FakeCDF']
        for type in d_type:
            if type == 'RealCDF':
                label = 0
            elif type == 'FakeCDF':
                label = 1
            dir_path = os.path.join(self.datapath, type)
            for file in os.listdir(dir_path):
                filepath = os.path.join(dir_path, file)
                self.vid_file_path.append((filepath, label))

        self.all_data = []
        self.all_idx_list = []
        self.get_all_vid()

        self.transform = preprocess

    def get_frames_per_vid(self, vid_file):
        frames_per_vid = []
        idx_list = []
        frame_count = len(os.listdir(vid_file))
        frame_idxs = np.linspace(0, frame_count - 1, self.n_frames, endpoint=True, dtype=int)

        for cnt_frame, img in enumerate(sorted(os.listdir(vid_file))):
            if cnt_frame not in frame_idxs:
                continue

            img_path = os.path.join(vid_file, img)
            idx_list_temp = [cnt_frame]
            idx_list += idx_list_temp
            frames_per_vid.append(img_path)
        return frames_per_vid, idx_list

    def get_all_vid(self):
        for vid_file, label in self.vid_file_path:
            frames_per_vid, idx_list = self.get_frames_per_vid(vid_file)
            self.all_data.append(frames_per_vid)
            self.all_idx_list.append(idx_list)
            self.label_list.append(label)

    def __getitem__(self, index):
        frames_per_vid = self.all_data[index]
        label = self.label_list[index]

        frames_concat = []
        for frame_path in frames_per_vid:
            frame = Image.open(frame_path).convert('RGB')
            # transform
            frame = self.transform(frame)
            frames_concat.append(frame)

        frames_concat = torch.stack(frames_concat, 0)
        return frames_concat, label

    def __len__(self):
        return len(self.all_data)

def set_dataset_singleGPU_CDF(config, preprocess, datapath='/data/CDF/faces/', n_frames=32):
    dataset = BuildCelebDFdataset_online(preprocess,  datapath, n_frames)
    print(f"successfully build CDF test dataset")

    data_loader_val = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )
    return dataset, data_loader_val

# ------------------------------------------------------------------------------------------------------------------- #
# for FaceShifter dataset
class BuildFShDataset_online(Dataset):
    def __init__(self, preprocess, datapath = '/data/FSh', n_frames=32, mode='c23'):
        super().__init__()

        self.n_frames = n_frames
        self.datapath_manip = os.path.join(datapath, mode, 'faces')
        self.datapath_real = os.path.join('data/FF++/', 'original_sequences/youtube', 'c23', 'faces')
        self.transform = preprocess

        self.vid_file_path = []
        self.label_list = []

        # for real path
        img_lines, _, _ = self.get_name_candi()
        for file in os.listdir(self.datapath_real):
            label = 0
            if file not in img_lines:
                continue
            filepath = os.path.join(self.datapath_real, file)
            self.vid_file_path.append((filepath, label))

        # for fake path
        for file in os.listdir(self.datapath_manip):
            label = 1
            if file not in img_lines:
                continue
            filepath = os.path.join(self.datapath_manip, file)
            self.vid_file_path.append((filepath, label))

        self.all_data = []
        self.all_idx_list = []
        self.get_all_vid()

    def get_name_candi(self):
        with open("data/FF++/test.json", 'r') as fd:
            data = json.load(fd)
            img_lines = []
            fake_lines = []
            real_lines = []
            for pair in data:
                r1, r2 = pair
                img_lines.append('{}'.format(r1))
                img_lines.append('{}'.format(r2))
                img_lines.append('{}_{}'.format(r1, r2))
                img_lines.append('{}_{}'.format(r2, r1))

                real_lines.append('{}'.format(r1))
                real_lines.append('{}'.format(r2))
                fake_lines.append('{}_{}'.format(r1, r2))
                fake_lines.append('{}_{}'.format(r2, r1))
        return img_lines, real_lines, fake_lines

    def get_all_vid(self):
        for vid_file, label in self.vid_file_path:
            frames_per_vid, idx_list = self.get_frames_per_vid(vid_file)
            self.all_data.append(frames_per_vid)
            self.all_idx_list.append(idx_list)
            self.label_list.append(label)

    def get_frames_per_vid(self, vid_file):
        frames_per_vid = []
        idx_list = []
        frame_count = len(os.listdir(vid_file))
        frame_idxs = np.linspace(0, frame_count - 1, self.n_frames, endpoint=True, dtype=int)

        for cnt_frame, img in enumerate(sorted(os.listdir(vid_file))):
            if cnt_frame not in frame_idxs:
                continue

            img_path = os.path.join(vid_file, img)
            idx_list_temp = [cnt_frame]
            idx_list += idx_list_temp
            frames_per_vid.append(img_path)
        return frames_per_vid, idx_list

    def __getitem__(self, index):
        frames_per_vid = self.all_data[index]
        label = self.label_list[index]

        frames_concat = []
        for frame_path in frames_per_vid:
            frame = Image.open(frame_path).convert('RGB')
            # transform
            frame = self.transform(frame)
            frames_concat.append(frame)

        frames_concat = torch.stack(frames_concat, 0)
        return frames_concat, label

    def __len__(self):
        return len(self.all_data)

def set_dataset_singleGPU_FSh(config, preprocess, datapath, n_frames, mode='c23'):
    dataset = BuildFShDataset_online(preprocess,  datapath, n_frames, mode)
    print(f"successfully build FaceShifter test dataset")
    print(f'length: {len(dataset)}')
    # pdb.set_trace()

    data_loader_val = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )
    return dataset, data_loader_val

# ------------------------------------------------------------------------------------------------------------------- #
# for DeeperForensics-1.0 dataset
class BuildDeeperDataset_online(Dataset):
    def __init__(self, preprocess, datapath='/data/DF-1.0/', n_frames=32, mode='end_to_end'):
        super().__init__()

        self.n_frames = n_frames
        self.datapath_manip = os.path.join(datapath, 'faces', mode)
        self.datapath_real = os.path.join('data/FF++/', 'original_sequences/youtube', 'c23', 'faces')
        self.transform = preprocess

        self.vid_file_path = []
        self.label_list = []

        # for real path
        img_lines, _, _ = self.get_name_candi()
        for file in os.listdir(self.datapath_real):
            label = 0
            if file not in img_lines:
                continue
            filepath = os.path.join(self.datapath_real, file)
            self.vid_file_path.append((filepath, label))

        # for fake path
        fake_tmp = []
        for file in os.listdir(self.datapath_manip):
            label = 1
            filepath = os.path.join(self.datapath_manip, file)
            fake_tmp.append((filepath, label))

        fake = random.sample(fake_tmp, len(self.vid_file_path))
        # fake = fake_tmp
        self.vid_file_path.extend(fake)


        self.all_data = []
        self.all_idx_list = []
        self.get_all_vid()

    def get_name_candi(self):
        with open("data/FF++/test.json", 'r') as fd:
            data = json.load(fd)
            img_lines = []
            fake_lines = []
            real_lines = []
            for pair in data:
                r1, r2 = pair
                img_lines.append('{}'.format(r1))
                img_lines.append('{}'.format(r2))
                img_lines.append('{}_{}'.format(r1, r2))
                img_lines.append('{}_{}'.format(r2, r1))

                real_lines.append('{}'.format(r1))
                real_lines.append('{}'.format(r2))
                fake_lines.append('{}_{}'.format(r1, r2))
                fake_lines.append('{}_{}'.format(r2, r1))
        return img_lines, real_lines, fake_lines

    def get_all_vid(self):
        for vid_file, label in self.vid_file_path:
            frames_per_vid, idx_list = self.get_frames_per_vid(vid_file)
            self.all_data.append(frames_per_vid)
            self.all_idx_list.append(idx_list)
            self.label_list.append(label)

    def get_frames_per_vid(self, vid_file):
        frames_per_vid = []
        idx_list = []
        frame_count = len(os.listdir(vid_file))
        frame_idxs = np.linspace(0, frame_count - 1, self.n_frames, endpoint=True, dtype=int)

        for cnt_frame, img in enumerate(sorted(os.listdir(vid_file))):
            if cnt_frame not in frame_idxs:
                continue

            img_path = os.path.join(vid_file, img)
            idx_list_temp = [cnt_frame]
            idx_list += idx_list_temp
            frames_per_vid.append(img_path)
        return frames_per_vid, idx_list

    def __getitem__(self, index):
        frames_per_vid = self.all_data[index]
        label = self.label_list[index]

        frames_concat = []
        for frame_path in frames_per_vid:
            frame = Image.open(frame_path).convert('RGB')
            # transform
            frame = self.transform(frame)
            frames_concat.append(frame)

        frames_concat = torch.stack(frames_concat, 0)
        return frames_concat, label

    def __len__(self):
        return len(self.all_data)


def set_dataset_singleGPU_Deeper(config, preprocess, datapath='/data/DF-1.0/',
                                 n_frames=32, mode='end_to_end'):
    dataset = BuildDeeperDataset_online(preprocess,  datapath, n_frames, mode)
    print(f"successfully build DeeperForensics-1.0 test dataset")
    print(f'length: {len(dataset)}')
    # pdb.set_trace()

    data_loader_val = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )
    return dataset, data_loader_val


# ------------------------------------------------------------------------------------------------------------------- #
# for test on DFDC dataset

def init_dfdc():
    label = pd.read_csv('/data/DFDC/labels.csv', delimiter=',')
    folder_list = [f'/data/DFDC/faces/{i}' for i in label['filename'].tolist()]
    label_list = label['label'].tolist()

    return folder_list, label_list

class BuildDFDCdataset_online(Dataset):
    def __init__(self, preprocess, n_frames=32):
        super().__init__()

        self.n_frames = n_frames
        self.transform = preprocess

        self.folder_list, self.target_list = init_dfdc()

        self.vid_file_path = []
        self.label_list = []
        self.all_data = []
        self.all_idx_list = []
        self.get_all_vid()

    def get_all_vid(self):
        for idx, vid_file_path in enumerate(self.folder_list):
            vid_file = vid_file_path.split('.')[0]
            if not os.path.exists(vid_file):
                continue
            length = len(os.listdir(vid_file))
            if length < self.n_frames:
                continue
            frames_per_vid, idx_list = self.get_frames_per_vid(vid_file)

            label = self.target_list[idx]

            self.all_data.append(frames_per_vid)
            self.all_idx_list.append(idx_list)
            self.label_list.append(label)

    def get_frames_per_vid(self, vid_file):
        frames_per_vid = []
        idx_list = []
        frame_count = len(os.listdir(vid_file))
        frame_idxs = np.linspace(0, frame_count - 1, self.n_frames, endpoint=True, dtype=int)

        for cnt_frame, img in enumerate(sorted(os.listdir(vid_file))):
            if cnt_frame not in frame_idxs:
                continue

            img_path = os.path.join(vid_file, img)
            idx_list_temp = [cnt_frame]
            idx_list += idx_list_temp
            frames_per_vid.append(img_path)
        return frames_per_vid, idx_list

    def __getitem__(self, index):
        frames_per_vid = self.all_data[index]
        label = self.label_list[index]

        frames_concat = []
        for frame_path in frames_per_vid:
            frame = Image.open(frame_path).convert('RGB')
            # transform
            frame = self.transform(frame)
            frames_concat.append(frame)

        frames_concat = torch.stack(frames_concat, 0)
        return frames_concat, label

    def __len__(self):
        return len(self.all_data)

def set_dataset_singleGPU_DFDC(config, preprocess, n_frames):
    dataset = BuildDFDCdataset_online(preprocess, n_frames)
    print(f"successfully build DFDC test dataset")
    print(f'length: {len(dataset)}')
    # pdb.set_trace()

    data_loader_val = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )
    return dataset, data_loader_val

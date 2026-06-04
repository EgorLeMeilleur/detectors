
import cv2, os, torch, json
import argparse
import subprocess
import cv2
from tqdm import tqdm
from os.path import join
import numpy as np
from face_utils2 import FaceDetector, norm_crop

config = {
    'test': "data/FF++/test.json"
}
train_mode = 'test'
with open(config[train_mode], 'r') as fd:
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

def extract_method_videos(video_paths, image_path, mode='c23'):
    for video in tqdm(os.listdir(video_paths)):
        video_name = video.split('.')[0]
        if video_name not in fake_lines:
            continue

        if mode == 'raw':
            video_path = join(video_paths, video)
            image_path_folder = join(image_path, video_name)
            os.makedirs(image_path_folder, exist_ok=True)
            extract_frames_faces_online(video_path, image_path_folder)
        elif mode == 'c23':
            if video_name not in img_lines:
                continue
            video_path = join(video_paths, video)
            image_path_folder = join(image_path, video_name)
            os.makedirs(image_path_folder, exist_ok=True)
            extract_frames_faces_online(video_path, image_path_folder)


def extract_frames_faces_online(data_path, output_path, method='cv2'):
    """Method to extract frames, either with ffmpeg or opencv. FFmpeg won't
    start from 0 so we would have to rename if we want to keep the filenames
    coherent."""
    os.makedirs(output_path, exist_ok=True)
    if method == 'ffmpeg':
        subprocess.check_output(
            'ffmpeg -i {} {}'.format(
                data_path, join(output_path, '%04d.png')),
            shell=True, stderr=subprocess.STDOUT)
    elif method == 'cv2':
        reader = cv2.VideoCapture(data_path)
        # # get the total num of frames
        # frame_num_total = reader.get(7)
        # ## set sample interval and return sample index
        # frame_idxs = np.linspace(0, int(frame_num_total) - 1, 64, endpoint=True, dtype=int)

        frame_num = 0
        while reader.isOpened():
            success, image = reader.read()
            if not success:
                break
            ############## extract face #################
            # if frame_num not in frame_idxs:
            #     frame_num += 1
            #     continue
            extract_faces(image, output_path, data_path, frame_num)
            #############################################
            frame_num += 1
        reader.release()
    else:
        raise Exception('Wrong extract frames method: {}'.format(method))


def extract_faces(image, output_path, datapath, frame_num):
    face_detector = FaceDetector()
    face_detector.load_checkpoint("RetinaFace-Resnet50-fixed.pth")
    try:
        boxes, landms = face_detector.detect(image)
    except AttributeError:
        print('AttributeError!' + datapath+' :'+str(frame_num))
    if boxes.shape[0] == 0:
        print('no faces detected in the current image!')
        return

    areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    max_face_idx = areas.argmax()
    landm = landms[max_face_idx]

    landmarks = landm.detach().numpy().reshape(5, 2).astype(np.int)
    img = norm_crop(image, landmarks, outsize=(317, 317))

    out_path = os.path.join(output_path, "%04d.png" % frame_num)
    cv2.imwrite(out_path, img)

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--video_path', type=str)
    p.add_argument('--save_path', type=str)
    args = p.parse_args()

    extract_method_videos(args.video_path, args.save_path)


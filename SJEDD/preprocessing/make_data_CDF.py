import pdb

import cv2, os, torch
import argparse
import subprocess
from tqdm import tqdm
from os.path import join
import numpy as np
from face_utils2 import FaceDetector, norm_crop

def extract_method_videos(video_list, image_path):
    label_list = []
    with open(video_list) as f:
        folder_list = []
        for data in f:
            line = data.split()
            path = line[1].split('/')
            folder_list += ['/data1/mianzou2/celeb-df/new-save-type/' + path[0] + '/videos/' + path[1]]
            label_list += [1 - int(line[0])]

    for i, video_path in enumerate(tqdm(folder_list)):
        video_name = video_path.split('/')[-1].split('.')[0]
        label = label_list[i]
        if label == 0:
            image_path_folder = join(image_path, 'RealCDF', video_name)
        elif label == 1:
            image_path_folder = join(image_path, 'FakeCDF', video_name)
        else:
            raise Exception('error on'+str(video_path))

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
        frame_num = 0
        while reader.isOpened():
            success, image = reader.read()
            if not success:
                break
            ############## extract face #################
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
    p.add_argument('--video_list', type=str, default="/data1/mianzou2/celeb-df/new-save-type/List_of_testing_videos.txt")
    p.add_argument('--save_path', type=str, default='/data0/mian2/celeb-df/dataset/')
    args = p.parse_args()

    extract_method_videos(args.video_list, args.save_path)
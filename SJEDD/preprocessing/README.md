## Preprocessing for video datasets
We first download the (test) datasets from the original project pages and put them into the ``data`` folder with the data structure as follows. Then, we run the following files to crop the faces for experiments:

### 1. Celeb-DF Dataset:
```
CUDA_VISIBLE_DEVICES=7 python make_data_CDF.py --video_list data/List_of_testing_videos.txt --save_path data/CDF/faces/
```
```
data
├── CDF
│   ├──Celeb-real
│   │   ├──videos
│   │   │   ├──id0_0001.mp4
│   │   │   ├──...
│   ├──Celeb-synthesis
│   │   ├──videos
│   │   │   ├──...
│   ├──YouTube-real
│   │   ├──videos
│   │   │   ├──...
│   ├──List_of_testing_videos.txt
│   ├──faces
│   │   ├──FakeCDF
│   │   │   ├──id0_id16_0003
│   │   │   │   ├──0000.png
│   │   │   │   ├──...
│   │   │   ├──id0_id16_0004
│   │   │   ├──...
│   │   ├──RealCDF
│   │   │   ├──00011
│   │   │   │   ├──0000.png
│   │   │   │   ├──...
│   │   │   ├──00021
│   │   │   ├──...
```

### 2. FaceShifter Dataset:
```
CUDA_VISIBLE_DEVICES=7 python make_data_FFpp_FSh.py --video_path data/FSh/c23/videos/ --save_path data/FSh/c23/faces/
```
```
data
├── FSh
│   ├──c23
│   │   ├──videos
│   │   │   ├──000_003.mp4
│   │   │   ├──...
│   │   ├──faces
│   │   │   ├──000_003
│   │   │   │   ├──0000.png
│   │   │   │   ├──...
│   │   │   ├──...
```

### 3. DeeperForensics-1.0 Dataset:
```
CUDA_VISIBLE_DEVICES=7 python make_data_DeeperFo.py --video_path data/DF-1.0/videos/ --save_path data/DF-1.0/faces/
```
```
data
├── DF-1.0
│   ├──videos
│   │   ├──end_to_end
│   │   │   ├──001_W101.mp4
│   │   │   ├──...
│   ├──faces
│   │   ├──end_to_end
│   │   │   ├──001_W101
│   │   │   │   ├──0000.png
│   │   │   │   ├──...
│   │   │   ├──...
```

### 4. DeepFakeDetectionChallenge Dataset:
```
CUDA_VISIBLE_DEVICES=7 python make_data_DFDC.py --video_path data/DFDC/videos/ --save_path data/DFDC/faces/
```
```
data
├── DFDC
│   ├──labels.csv
│   ├──videos
│   │   ├──aalscayrfi.mp4
│   │   ├──...
│   ├──faces
│   │   ├──aalscayrfi
│   │   │   ├──0000.png
│   │   │   ├──...
```

### 5. FaceForensics++ (FF++) Dataset:
```
original: CUDA_VISIBLE_DEVICES=7 python make_data_FFpp_FSh.py --video_path data/FF++/original_sequences/youtube/c23/videos/ --save_path data/FF++/original_sequences/youtube/c23/faces/
Deepfakes: CUDA_VISIBLE_DEVICES=7 python make_data_FFpp_FSh.py --video_path data/FF++/manipulated_sequences/Deepfakes/c23/videos/ --save_path data/FF++/manipulated_sequences/Deepfakes/c23/faces/
Face2Face: CUDA_VISIBLE_DEVICES=7 python make_data_FFpp_FSh.py --video_path data/FF++/manipulated_sequences/Face2Face/c23/videos/ --save_path data/FF++/manipulated_sequences/Face2Face/c23/faces/
FaceSwap: CUDA_VISIBLE_DEVICES=7 python make_data_FFpp_FSh.py --video_path data/FF++/manipulated_sequences/FaceSwap/c23/videos/ --save_path data/FF++/manipulated_sequences/FaceSwap/c23/faces/
NeuralTextures: CUDA_VISIBLE_DEVICES=7 python make_data_FFpp_FSh.py --video_path data/FF++/manipulated_sequences/NeuralTextures/c23/videos/ --save_path data/FF++/manipulated_sequences/NeuralTextures/c23/faces/
```
```
data
├── FF++
│   ├──train.json
│   ├──val.json
│   ├──test.json
│   ├──original_sequences
│   │   ├──youtube
│   │   │   ├──c23
│   │   │   │   ├──videos
│   │   │   │   │   ├──000.mp4
│   │   │   │   │   ├──...
│   │   │   │   ├──faces
│   │   │   │   │   ├──000
│   │   │   │   │   │   ├──000.png
│   │   │   │   │   │   ├──...
│   ├──manipulated_sequences
│   │   ├──Deepfakes
│   │   │   ├──c23
│   │   │   │   ├──videos
│   │   │   │   │   ├──000_003.mp4
│   │   │   │   │   ├──...
│   │   │   │   ├──faces
│   │   │   │   │   ├──000_003
│   │   │   │   │   │   ├──000.png
│   │   │   │   │   │   ├──...
│   │   ├──Face2Face
│   │   ├──FaceSwap
│   │   ├──NeuralTextures
```

### 6. DiFF Dataset:
```
data
├── DiFF
│   ├──DiFFtest.txt
│   ├──DiFF_real
│   ├──DCFace
│   ├──DiffFace
```

### 7. DiffusionFace Dataset:
```
data
├── DiffusionFace
│   ├──DiffusionFacetest.txt
│   ├──Real
│   ├──DiffSwap
```

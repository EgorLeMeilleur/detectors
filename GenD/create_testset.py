import os
import random
import shutil

fake_dir = "/mnt/tank/scratch/dstoronkin/celebdf/crop_img"
real_dir = "/mnt/tank/scratch/dstoronkin/celebdf/real"

fake_new_dir = "/mnt/tank/scratch/dstoronkin/sjedd/celebdf_download/fake"
real_new_dir = "/mnt/tank/scratch/dstoronkin/sjedd/celebdf_download/real"

os.makedirs(fake_new_dir)
os.makedirs(real_new_dir)

N_FAKE = 3000
N_REAL = 1000

random.seed(168)

fake_imgs = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)]
real_imgs = [os.path.join(real_dir, f) for f in os.listdir(real_dir)]

fake_sel = random.sample(fake_imgs, N_FAKE)
real_sel = random.sample(real_imgs, N_REAL)

for file in fake_sel:
    shutil.copy(file, os.path.join(fake_new_dir, os.path.basename(file)))

for file in real_sel:
    shutil.copy(file, os.path.join(real_new_dir, os.path.basename(file)))
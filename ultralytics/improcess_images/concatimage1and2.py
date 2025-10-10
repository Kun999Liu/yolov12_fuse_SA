import numpy as np
import glob
import os

# from ultralytics.data.utils import readTif
from pathlib import Path
from tqdm import tqdm
from ultralytics.data.utils import readTif

# npy_files = glob.glob('datasets/mydata/images/train/*.npy')
# print(npy_files)

# current_folder = 'datasets/mydata/images/train'
# another_folder = 'datasets/mydata/images2/train'
# "C:\Users\liuku\Desktop\datasets"

current_folder = r"C:\Users\liuku\Desktop\datasets\image1"
another_folder = r"C:\Users\liuku\Desktop\datasets\image2"

tif_files = glob.glob(os.path.join(current_folder, '*.tif'))
npy_files = glob.glob(os.path.join(current_folder, '*.npy'))

matching_files = []
for tif_file in tqdm(tif_files):
    base_name = os.path.splitext(os.path.basename(tif_file))[0]
    npy_file = os.path.join(current_folder, base_name + '.npy')
    # im_width1, im_height1, im_bands1, projection1, geotrans1, im1 = readTif(tif_file)
    im1 = readTif(tif_file)
    print(im1.shape)
    # np.save(Path(npy_file).as_posix(), im1, allow_pickle=False)

    # base_name = os.path.splitext(os.path.basename(tif_file))[0]
    target_file = os.path.join(another_folder, base_name + '.npy')
    # im_width, im_height, im_bands, projection, geotrans, im2 = readTif(os.path.join(another_folder, base_name+'.tif'))
    im2 = readTif(os.path.join(another_folder, base_name + '.tif'))
    # print("tif影像：", im2, "tif大小：", im2.shape)
    # np.save(Path(target_file).as_posix(), im2, allow_pickle=False)
    matching_files.extend(glob.glob(target_file))

    result = np.concatenate((im1, im2, im2, im2), axis=-1)
    np.save(Path(npy_file).as_posix(), result, allow_pickle=False)

    print("npy_file:", npy_file, '\n', 'shape of result:', result.shape)

# for file in matching_files:
#     print(file)

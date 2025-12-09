import ultralytics
print(ultralytics.__version__)
print(ultralytics.__file__)

from ultralytics.data.utils import read_image
img = read_image(r"F:\wxy_code\mydata_7bands\images\train\GF2_PMS1_E82.7_N45.2_20220430_L1A0006441524-pansharpen2colNum_12rowNum_39.npy", "npy")
print(img.shape[0], img.shape[1], img.shape[2])
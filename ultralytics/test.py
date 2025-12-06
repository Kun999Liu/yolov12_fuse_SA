import ultralytics
print(ultralytics.__version__)
print(ultralytics.__file__)

from ultralytics.data.utils import read_image
img = read_image(r"D:\windfram&tower\transmission_7bands\images_2\train\images\GF2_PMS1_E93.5_N42.6_20250624_L1A14721219001_fuse_1_52.npy", "npy")
print(img.shape[0], img.shape[1], img.shape[2])
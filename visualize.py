import numpy as np
import matplotlib.pyplot as plt
import os
from tkinter import filedialog

"""
This script takes a directory of depth images (numpy arrays) and saves them as jpgs.
Note: can take up to 2.5 mins for ~200 images.
"""
make_videos = True


# Prompt for directory of depths
data_dir = "/home/nate/Documents/simplerecon/tello/tello-fusion-images-2023-05-13-12-18-27"
assert os.path.exists(data_dir), "Data Directory does not exist."
# calculated depth directory (not empty)
depth_dir = "/".join(data_dir.split("/")[:-1]) + "/depths-from-" + data_dir.split("/")[-1]
assert os.path.exists(depth_dir), "Depth Directory does not exist."
# depth image directory (empty)
depth_img_dir = "/".join(depth_dir.split("/")[:-1]) + "/depth-imgs-from-" + depth_dir.split("/")[-1]
if not os.path.exists(depth_img_dir):
    os.makedirs(depth_img_dir)

# for each file in depth_dir, load depth and save as jpg
for file in sorted(os.listdir(depth_dir)):
    print(file)
    # load depth
    depth = np.load(depth_dir + "/" + file) # numpy array
    # save to jpg
    plt.imshow(depth); plt.savefig(depth_img_dir + "/" + file + ".jpg")



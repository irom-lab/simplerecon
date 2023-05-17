import cv2
import os

"""
Short script to convert a directory of images to a video.
Converts depth and raw rgb images to a video mp4.
"""

# Prompt for directory of depths
data_dir = "/home/nate/Documents/simplerecon/tello/tello-fusion-images-2023-05-13-12-18-27"
assert os.path.exists(data_dir), "Data Directory does not exist."
# calculated depth directory (not empty)
depth_dir = "/".join(data_dir.split("/")[:-1]) + "/depths-from-" + data_dir.split("/")[-1]
assert os.path.exists(depth_dir), "Depth Directory does not exist."
# depth image directory (empty)
depth_img_dir = "/".join(depth_dir.split("/")[:-1]) + "/depth-imgs-from-" + depth_dir.split("/")[-1]
assert os.path.exists(depth_img_dir), "Depth Image Directory does not exist."


def make_video(image_dir, video_name, fps=10):
    images = [img for img in os.listdir(image_dir) if img.endswith('.jpg')]
    images.sort()  # sort the list of image files

    frame = cv2.imread(os.path.join(image_dir, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, 10, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_dir, image)))

    cv2.destroyAllWindows()
    video.release()


# Make videos from source and depth images.

rgb_video_path = 'tello/videos/rgb_video.mp4'
depth_video_path = 'tello/videos/depth_video.mp4'
fps = 30  # Frames per second of the output video

make_video(data_dir, rgb_video_path, fps)
make_video(depth_img_dir, depth_video_path, fps)
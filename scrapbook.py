from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

file = "/home/nsimon/Documents/MonoNav/synchronized-trial-2023-06-24-22-50-32/tello-depth-images-sr/tello_frame-000007.depth.npy"
depth = np.load(file)
plt.imshow(depth); plt.show()

im = Image.fromarray(depth)
im_upscaled = im.resize((960, 720), Image.BILINEAR)

# Convert PIL image to NumPy array
numpy_image = np.array(im_upscaled)

# Display the image using plt.show
plt.imshow(numpy_image)
plt.show()
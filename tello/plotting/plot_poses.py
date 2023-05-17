import numpy as np
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import matplotlib.animation as animation


#####################################################################

current_directory = os.getcwd()
print("Current directory:", current_directory)

# Specify the directory with the images
data_dir = "/home/nate/Documents/simplerecon/tello/tello-fusion-images-2023-05-13-12-18-27"
poses = [pose for pose in os.listdir(data_dir) if pose.endswith('.txt')]
poses.sort()  # sort the list of image files

# # optional: use a subset of images
# start_index = poses.index('frame-000100.pose.txt')
# end_index = poses.index('frame-000163.pose.txt')
# poses = poses[start_index:end_index+1]

# Initialize arrays
x_vals = []
y_vals = []
z_vals = []

# Load data from files
for pose in poses:
    cam_pose = np.loadtxt(data_dir + "/" + pose)
    x_vals.append(cam_pose[2, 3])
    y_vals.append(cam_pose[0, 3])
    z_vals.append(cam_pose[1, 3])

x_vals = [x - x_vals[0] for x in x_vals]
y_vals = [y - y_vals[0] for y in y_vals]
z_vals = [z - z_vals[0] for z in z_vals]

# Plot data
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
num_poses = len(x_vals)
for i in range(num_poses-1):
    ax.plot(x_vals[i:i+2], y_vals[i:i+2], z_vals[i:i+2], c=(i/(num_poses-1), 0, (num_poses-i-1)/(num_poses-1)))
ax.scatter(x_vals[0], y_vals[0], z_vals[0], c='b', label='start')
ax.scatter(x_vals[-1], y_vals[-1], z_vals[-1], c='r', label='end')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('Camera Poses')
ax.legend()
plt.savefig('camera_poses.png')

# Create color map
colors = cm.rainbow(np.linspace(0, 1, len(x_vals)))

# Plot data
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# # Create initial plot
# point, = ax.plot(x_vals, y_vals, z_vals, color=colors[0], label='start')
# ax.scatter(x_vals, y_vals, z_vals, c=colors, alpha=0.5)

# # Set axis labels and title
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.set_title('Camera Poses')

# # Create update function for animation
# def update(i):
#     point.set_data(x_vals[:i+1], y_vals[:i+1])
#     point.set_3d_properties(z_vals[:i+1])
#     point.set_color(colors[i])
#     if i == len(x_vals)-1:
#         point.set_label('end')
#         ax.legend()
#     return point,

# # Create animation
# anim = FuncAnimation(fig, update, frames=len(x_vals), interval=100, blit=True)

# # Save animation as mp4
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
# anim.save('camera_poses.mp4', writer=writer)
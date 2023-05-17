""" 
Simple script for testing SimpleRecon.
Takes a directory of images+poses and computes a depth map based on overlapping adjacent sets of 7 images+poses.
Integrates the volume by way of method in https://github.com/irom-lab/tello-zoedepth/blob/main/fuse_tello_images.py
to produce .pcd and .mesh files.
"""

import torch
import torch.nn.functional as F


from experiment_modules.depth_model import DepthModel
import options
from utils.generic_utils import read_image_file
from utils.generic_utils import to_gpu
from tools import fusers_helper


from PIL import Image
import numpy as np
from tkinter import filedialog

import os
import open3d as o3d
from PIL import Image

def print_image_info(image):
    print("width =", image.width)
    print("height =", image.height)
    print("channels =", image.num_of_channels)
    print("bytes per channel =", image.bytes_per_channel)

def main(opts):

    ########################################################################################
    # Set up data

    # data_dir = filedialog.askdirectory(title="Double click on directory with images and open") #prompt user
    data_dir = "/home/nate/Documents/simplerecon/tello/tello-fusion-images-2023-05-13-12-18-27"
    print("Chosen directory: ", data_dir)
    depth_dir = "/".join(data_dir.split("/")[:-1]) + "/depths-from-" + data_dir.split("/")[-1]
    mesh_output_dir = "/".join(data_dir.split("/")[:-1]) + "/mesh-from-" + data_dir.split("/")[-1]
    # Example:
    # data_dir = "/home/nate/Documents/simplerecon/tello/tello-fusion-images-2023-05-13-12-18-27"
    # depth_dir = "/home/nate/Documents/simplerecon/tello/depths-from-tello-fusion-images-2023-05-13-12-18-27"

    # if it does not exist, make depth_dir and mesh_output_dir
    if not os.path.exists(depth_dir):
        os.makedirs(depth_dir)
    if not os.path.exists(mesh_output_dir):
        os.makedirs(mesh_output_dir)
    
    # Determine number of images and poses, and assert that they are the same. (In actual stream, this is not necessary.)
    # Number of files ending with .color.jpg in data_dir
    num_imgs = len([name for name in os.listdir(data_dir) if name.endswith(".color.jpg")])
    print("Number of images: ", num_imgs)
    # Number of files ending with .color.jpg in data_dir
    num_poses = len([name for name in os.listdir(data_dir) if name.endswith(".pose.txt")])
    print("Number of poses: ", num_poses)
    assert num_poses == num_imgs, "Number of images and poses must be the same."

    # Specify the number of source images to use
    num_src_imgs = 7 #n images to use for the n+1st image's depth estimate

    ##################################### LOOP-INDEPENDENT ITEMS #####################################
    # Current image #TODO: check this; I'm seeing 960x720?
    img_height = 384
    img_width = 512

    # Tello camera intrinsics
    K_tello = np.array([[929.562627, 0, 487.474037, 0], [0, 928.604856, 363.165223, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) 

    # Rescale to compute K_s1_b44
    # TODO: Triple check that this is correct; see load_intrinsics in https://github.com/nianticlabs/simplerecon/blob/3a74095f459dce62579348de51e78493d9ec88eb/datasets/vdr_dataset.py#L207-L226
    # It looks like K_si_b44 is computed by dividing K by 2^i, where i is the scale index
    #TODO: Double check that there is no other scaling that needs to be applied (e.g., due to change in resolution). I don't think so, but not 100% sure.
    #NATE: Why are we only calculating K_s1_b44, as opposed to i in range(5)?
    K_tello[:2] /= 2 

    # Compute inverse
    cur_invK = np.linalg.inv(K_tello)

    # Add batch dimension
    cur_K = K_tello[np.newaxis, :, :] # Add batch dimension
    cur_invK = cur_invK[np.newaxis, :, :] # Add batch dimension

    # Source images
    src_imgs_b3hw = torch.zeros((1,num_src_imgs,3,img_height, img_width), dtype=torch.float32)
    src_K_s1_b44 = np.zeros([1,num_src_imgs,4,4])
    src_cam_T_world_b44 = np.zeros([1,num_src_imgs,4,4])
    src_world_T_cam_b44 = np.zeros([1,num_src_imgs,4,4])


    #####################################################################
    # Define hyperparameters for TSDF fusion
    voxel_length = 0.02
    sdf_trunc = 0.1
    depth_trunc = 10.0

    # Initialize TSD volume
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    
    # Create o3d camera #TODO: compare with K_tello
    camera = o3d.camera.PinholeCameraIntrinsic(960, 720, 929.562627, 928.604856, 487.474037, 363.165223) # (self, width, height, fx, fy, cx, cy)
    #####################################################################

    ##################################### LOOP OVER IMAGES+POSES #####################################
    #Note: This is perhaps not being the most efficient with the first and last 7 images.
    for cur_img_ind in range(num_src_imgs, num_imgs):
        # Specify indices for the source images
        src_img_inds = range(cur_img_ind-num_src_imgs,cur_img_ind) # look n images back from current image

        cur_img = read_image_file(data_dir + "/frame-%06d.color.jpg"%(cur_img_ind), height=img_height, width=img_width) # Load image (cool zero padding trick)
        # img_height = cur_img.shape[1]
        # img_width = cur_img.shape[2]
        cur_img = cur_img[np.newaxis, :, :, :] # Add batch dimension

        # Get pose of current image
        cur_pose = np.loadtxt(data_dir + "/frame-%06d.pose.txt"%(cur_img_ind)) # Load pose

        cur_cam_T_world = cur_pose # TODO: Triple check that cur_pose is in fact cam_T_world (and not the other way around!)
        cur_world_T_cam = np.linalg.inv(cur_cam_T_world) 

        cur_cam_T_world = cur_cam_T_world[np.newaxis, :, :] # Add batch dimension
        cur_world_T_cam = cur_world_T_cam[np.newaxis, :, :] # Add batch dimension

        # Setup data dictionary for current image as torch tensors
        cur_data = {
            "image_b3hw": cur_img, 
            "K_s1_b44": torch.from_numpy(cur_K).float(),
            "invK_s1_b44": torch.from_numpy(cur_invK).float(),
            "cam_T_world_b44": torch.from_numpy(cur_cam_T_world).float(),
            "world_T_cam_b44": torch.from_numpy(cur_world_T_cam).float(),
        }

        # Transfer to GPU
        cur_data = to_gpu(cur_data)

        ########################################################################################
        #TODO: Check that this can be performed once for the entire session.
        # Set up model. Note that we're not passing in opts as an argument, although
        # we could. We're being pretty stubborn with using the options the model had
        # used when training, saved internally as part of hparams in the checkpoint.
        # You can change this at inference by passing in 'opts=opts,' but there 
        # be dragons if you're not careful.
        model = DepthModel.load_from_checkpoint(
                                    opts.load_weights_from_checkpoint,
                                    args=None)

        model.cost_volume = model.cost_volume.to_fast()
        model = model.cuda().eval()
        ########################################################################################

        #####################################

        #####################################

        ind = 0 # Index for source images #TODO: check that this is correct in the moving window case (Nate thinks it is)
        for src_img_ind in src_img_inds:
            # Load image
            src_img = read_image_file(data_dir + "/frame-%06d.color.jpg"%(src_img_ind),height=img_height, width=img_width) # Load image
            src_imgs_b3hw[0,ind,:,:,:] = src_img

            # Intrinsics
            src_K_s1_b44[0,ind,:,:] = K_tello

            # Load pose
            src_pose = np.loadtxt(data_dir + "/frame-%06d.pose.txt"%(src_img_ind)) # Load pose
            src_cam_T_world_b44[0,ind,:,:] = src_pose # TODO: Again, double check that this is correct
            src_world_T_cam_b44[0,ind,:,:] = np.linalg.inv(src_pose)

            ind += 1

        # Setup data dictionary for source images as torch tensors
        src_data = {
            "image_b3hw": src_imgs_b3hw, 
            "K_s1_b44": torch.from_numpy(src_K_s1_b44).float(),
            "cam_T_world_b44": torch.from_numpy(src_cam_T_world_b44).float(),
            "world_T_cam_b44": torch.from_numpy(src_world_T_cam_b44).float(),
        }

        # Transfer to GPU
        src_data = to_gpu(src_data)

        ########################################################################################


        ########################################################################################
        # Run inference
        outputs = model(
                    "test", cur_data, src_data, 
                    unbatched_matching_encoder_forward=(
                        not opts.fast_cost_volume
                    ), 
                    return_mask=True,
                )
        ########################################################################################

        ########################################################################################
        # Get depth
        depth = outputs["depth_pred_s0_b1hw"] # TODO: Make sure this is actually depth we want; there are other outputs too. 

        # Save depth image
        depth = depth[0,0,:,:].detach().cpu().numpy()
        depth = depth * 1000 # Convert to mm
        # depth_PIL = Image.fromarray(depth)
        # depth_PIL.save("./depth.millimeters.frame-%06d.jpg"%(cur_img_ind))

        # Save depth as numpy array
        np.save(depth_dir+"/depth.millimeters.frame-%06d.npy"%(cur_img_ind), depth)

        # Print done
        print(" ")
        print("Done computing depth! Saved to ./depth.millimeters.frame-%06d.npy"%(cur_img_ind))
        print(" ")

        ########################################################################################
        
        # Fusion - Ani's ZoeDepth Approach

        # Example usage
        # upsample to match the image size
        upsampled_depth_pred_b1hw = F.interpolate(
                                    outputs["depth_pred_s0_b1hw"], 
                                    size=(720, 960), #TODO: define and use img_height and img_width instead
                                    mode="nearest",
                                )
        depth_highres = upsampled_depth_pred_b1hw[0,0,:,:].detach().cpu().numpy()
        depth_highres = depth_highres * 1000 # Convert to mm

        # Convert image and depth to o3d
        color = Image.open(data_dir + "/frame-%06d.color.jpg"%(cur_img_ind)).convert("RGB") # Load image
        color_o3d = o3d.geometry.Image(np.asarray(color)) # Convert to open3d image

        depth_uint16 = depth_highres.astype(np.uint16) # Convert to uint16
        depth_o3d = o3d.geometry.Image(depth_uint16) # Convert to open3d image

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_o3d, depth_o3d, depth_trunc=depth_trunc, convert_rgb_to_intensity=False) 

        # Integrate
        volume.integrate(rgbd,camera,np.linalg.inv(cur_pose))
    
    #####################################################################
    # Extract point cloud
    pcd = volume.extract_point_cloud()
    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(mesh_output_dir + "/tello-fusion-pcd.ply", pcd)
    #####################################################################

    #####################################################################
    # Extract mesh
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])
    o3d.io.write_triangle_mesh(mesh_output_dir + "/tello-fusion-mesh.ply", mesh)
    #####################################################################

if __name__ == '__main__':
    # don't need grad for test.
    torch.set_grad_enabled(False)

    # get an instance of options and load it with config file(s) and cli args.
    option_handler = options.OptionsHandler()
    option_handler.parse_and_merge_options()
    option_handler.pretty_print_options()
    print("\n")
    opts = option_handler.options

    # if no GPUs are available for us then, use the 32 bit on CPU
    if opts.gpus == 0:
        print("Setting precision to 32 bits since --gpus is set to 0.")
        opts.precision = 32

    main(opts)

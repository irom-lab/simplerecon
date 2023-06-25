""" 
Takes a directory of images+poses and computes a depth map based on overlapping adjacent sets of 7 images+poses.
It then fuses the depth maps into a live video copying the approach from visualize_live_meshing.py
Videos end up in tello/videos and currently have problems.
"""

import torch
import torch.nn.functional as F

from experiment_modules.depth_model import DepthModel
import options
from utils.generic_utils import read_image_file
from utils.generic_utils import to_gpu
from tools import fusers_helper
from tools.mesh_renderer import (DEFAULT_CAM_FRUSTUM_MATERIAL,
                                 DEFAULT_MESH_MATERIAL, Renderer,
                                 SmoothBirdsEyeCamera, camera_marker,
                                 create_light_array, get_image_box,
                                 transform_trimesh)
from utils.dataset_utils import get_dataset
from utils.visualization_utils import colormap_image, save_viz_video_frames

from PIL import Image
import numpy as np
import pyrender
import trimesh
from tkinter import filedialog

import os
import open3d as o3d
from pathlib import Path


fpv_renderer = Renderer(height=192, width=256)
birdseye_renderer = Renderer(height=192, width=256)

# rendering and fusion presets
smooth_birdseye = SmoothBirdsEyeCamera()
mesh_render_fpv_frames = []
mesh_render_birdeye_frames = []

video_output_dir = "/home/nate/Documents/simplerecon/tello/videos"

def main(opts):
    fuser = fusers_helper.get_fuser(opts, None)

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
        depth_pred = outputs["depth_pred_s0_b1hw"] # TODO: Make sure this is actually depth we want; there are other outputs too. 

        # Save depth image
        depth = depth_pred[0,0,:,:].detach().cpu().numpy()
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
        # Create mesh!!

        color_frame = (cur_data["high_res_color_b3hw"] 
                            if  "high_res_color_b3hw" in cur_data 
                            else cur_data["image_b3hw"])
        fuser.fuse_frames(depth_pred, cur_data["K_s1_b44"], #original: K_s0_b44
                                    cur_data["cam_T_world_b44"], 
                                    color_frame)

        mesh_path=os.path.join(mesh_output_dir,f"{cur_img_ind}.ply")
        fuser.export_mesh(path=mesh_path)

        trimesh_path = mesh_path

        if opts.fuse_color:
            scene_trimesh_mesh = trimesh.load(trimesh_path, 
                                                    force='mesh')
        else:
            scene_trimesh_mesh = fuser.get_mesh(
                                            convert_to_trimesh=True)


        ########################################################################################
        # Get render!!
        
        world_T_cam_44 = cur_data["world_T_cam_b44"].squeeze().cpu().numpy()
        K_33 = cur_data["K_s1_b44"].squeeze().cpu().numpy() #original: K_s0_b44

        render_height = opts.viz_render_height
        render_width = opts.viz_render_width
        K_33[0] *= (render_width/depth_pred.shape[-1]) 
        K_33[1] *= (render_height/depth_pred.shape[-2])

        light_pos = world_T_cam_44.copy()
        light_pos[2, 3] += 5.0
        lights = create_light_array(
                            pyrender.PointLight(intensity=30.0), 
                            light_pos, 
                            x_length=12, 
                            y_length=12, 
                            num_x=6, 
                            num_y=6,
                        )
        meshes = ([] if scene_trimesh_mesh is None 
                                            else [scene_trimesh_mesh])
        render_fpv = fpv_renderer.render_mesh(
                                    meshes,   
                                    render_height, render_width, 
                                    world_T_cam_44, K_33, 
                                    True, 
                                    lights=lights,
                                )

        meshes = ([] if scene_trimesh_mesh is None 
                                            else [scene_trimesh_mesh])
        mesh_materials = ([None] if opts.fuse_color 
                                        else [DEFAULT_MESH_MATERIAL])

        fpv_camera = trimesh.scene.Camera(
                            resolution=(render_height, render_width), 
                            focal=(K_33[0][0], K_33[1][1])
                        )

        cam_marker_size = 0.7
        cam_marker_mesh = camera_marker(fpv_camera, 
                                    cam_marker_size=cam_marker_size)[1]

        np_vertices = np.array(cam_marker_mesh.vertices)

        np_vertices = (world_T_cam_44 @ np.concatenate([np_vertices, 
                            np.ones((np_vertices.shape[0], 1))], 1).T).T

        np_vertices = np_vertices/np_vertices[:,3][:,None]
        cam_marker_mesh = trimesh.Trimesh(vertices=np_vertices[:,:3], 
                                            faces=cam_marker_mesh.faces)

        meshes.append(cam_marker_mesh)
        mesh_materials.append(DEFAULT_CAM_FRUSTUM_MATERIAL)

        our_depth_3hw = colormap_image(depth_pred.squeeze(0), 
                                            vmin=0, vmax=3.0)
        our_depth_hw3 = our_depth_3hw.permute(1,2,0)
        pil_depth = Image.fromarray(
                        np.uint8(
                            our_depth_hw3.cpu().detach().numpy() * 255))

        image_mesh = get_image_box(
                            pil_depth, 
                            cam_marker_size=cam_marker_size,
                            fovs=(fpv_camera.fov[0], fpv_camera.fov[1])
                        )

        image_mesh = transform_trimesh(image_mesh, world_T_cam_44)
        meshes.append(image_mesh)
        mesh_materials.append(None)

        image_mesh = get_image_box(
                            pil_depth, 
                            cam_marker_size=cam_marker_size, 
                            flip=True,
                            fovs=(fpv_camera.fov[0], fpv_camera.fov[1])
                        )

        image_mesh.vertices[:,2] += 0.01
        image_mesh = transform_trimesh(image_mesh, world_T_cam_44)
        meshes.append(image_mesh)
        mesh_materials.append(None)

        birdeye_world_T_cam_44 = smooth_birdseye.get_bird_eye_trans(
                                    scene_trimesh_mesh, 
                                    fpv_pose=world_T_cam_44
                                )
        
        if opts.back_face_alpha:
            render_birdseye = birdseye_renderer.render_mesh_cull_composite(
                                    meshes=meshes, 
                                    height=render_height,
                                    width=render_width, 
                                    world_T_cam=birdeye_world_T_cam_44, 
                                    K=K_33, 
                                    get_colour=True, 
                                    mesh_materials=mesh_materials, 
                                    lights=lights, 
                                    alpha=opts.back_face_alpha,
                                )
        else:
            render_birdseye = birdseye_renderer.render_mesh(
                            meshes, 
                            render_height, render_width, 
                            birdeye_world_T_cam_44, 
                            K_33, True, mesh_materials=mesh_materials, 
                            lights=lights,
                        )


        mesh_render_fpv_frames.append(render_fpv)
        mesh_render_birdeye_frames.append(render_birdseye)
    
    fps = (opts.standard_fps if opts.skip_frames is None 
                        else round(opts.standard_fps/opts.skip_frames))
    
    #print("mesh_render_fpv_frames",mesh_render_fpv_frames) #TODO: fix empty
    save_viz_video_frames(mesh_render_fpv_frames, 
                    video_output_dir + "_fpv.mp4", fps=fps)
    save_viz_video_frames(mesh_render_birdeye_frames, 
                    video_output_dir + "_birdseye.mp4", fps=fps)


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

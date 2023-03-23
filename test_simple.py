""" 
Simple script for testing SimpleRecon.
"""

import torch

from experiment_modules.depth_model import DepthModel
import options
from utils.generic_utils import read_image_file
from utils.generic_utils import to_gpu

from PIL import Image
import numpy as np
from tkinter import filedialog


def main(opts):

    ########################################################################################
    # Set up data

    data_dir = filedialog.askdirectory(title="Double click on directory with images and open")
    print("Chosen directory: ", data_dir)

    # Specify indices for the current and source images
    cur_img_ind = 7 # 8th image as current image
    src_img_inds = range(0,7) # 1st to 7th image as source images

    #####################################
    # Current image
    img_height = 384
    img_width = 512
    cur_img = read_image_file(data_dir + "/frame-%06d.color.jpg"%(cur_img_ind), height=img_height, width=img_width) # Load image
    # img_height = cur_img.shape[1]
    # img_width = cur_img.shape[2]
    cur_img = cur_img[np.newaxis, :, :, :] # Add batch dimension

    # Tello camera intrinsics
    K_tello = np.array([[929.562627, 0, 487.474037, 0], [0, 928.604856, 363.165223, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) 

    # Rescale to compute K_s1_b44
    # TODO: Triple check that this is correct; see load_intrinsics in https://github.com/nianticlabs/simplerecon/blob/3a74095f459dce62579348de51e78493d9ec88eb/datasets/vdr_dataset.py
    # It looks like K_si_b44 is computed by dividing K by 2^i, where i is the scale index
    #TODO: Double check that there is no other scaling that needs to be applied (e.g., due to change in resolution). I don't think so, but not 100% sure.
    K_tello[:2] /= 2 

    # Compute inverse
    cur_invK = np.linalg.inv(K_tello)

    # Add batch dimension
    cur_K = K_tello[np.newaxis, :, :] # Add batch dimension
    cur_invK = cur_invK[np.newaxis, :, :] # Add batch dimension

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

    #####################################

    #####################################
    # Source images
    num_src_imgs = len(src_img_inds)
    src_imgs_b3hw = torch.zeros((1,num_src_imgs,3,img_height, img_width), dtype=torch.float32)
    src_K_s1_b44 = np.zeros([1,num_src_imgs,4,4])
    src_cam_T_world_b44 = np.zeros([1,num_src_imgs,4,4])
    src_world_T_cam_b44 = np.zeros([1,num_src_imgs,4,4])

    ind = 0 # Index for source images
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
    np.save("./depth.millimeters.frame-%06d.npy"%(cur_img_ind), depth)

    # Print done
    print(" ")
    print("Done computing depth! Saved to ./depth.millimeters.frame-%06d.npy"%(cur_img_ind))
    print(" ")


    ########################################################################################
    
    # # Fusion
    # upsampled_depth_pred_b1hw = F.interpolate(
    #                             outputs["depth_pred_s0_b1hw"], 
    #                             size=(depth_gt.shape[-2], depth_gt.shape[-1]),
    #                             mode="nearest",
    #                         )
    # color_frame = (cur_data["high_res_color_b3hw"] 
    #             if  "high_res_color_b3hw" in cur_data 
    #                 else cur_data["image_b3hw"])
    # fuser.fuse_frames(
    #                 upsampled_depth_pred_b1hw, 
    #                 cur_data["K_full_depth_b44"], 
    #                 cur_data["cam_T_world_b44"], 
    #                 color_frame
    #         )
    # fuser.export_mesh(
    #                 os.path.join(mesh_output_dir, 
    #                     f"{scan.replace('/', '_')}.ply"),
    #             )

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

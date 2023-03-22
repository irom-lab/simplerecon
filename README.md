# SimpleRecon: 3D Reconstruction Without 3D Convolutions

Fork of SimpleRecon that includes a very simple test script. See [here](https://github.com/nianticlabs/simplerecon) for the full README. See below for instructions on how to run test_simple.py in this repo. 

## ⚙️ Setup

Assuming a fresh [Anaconda](https://www.anaconda.com/download/) distribution, you can install dependencies with:
```shell
conda env create -f simplerecon_env.yml
```
We ran our experiments with PyTorch 1.10, CUDA 11.3, Python 3.9.7 and Debian GNU/Linux 10.

## 📦 Models

Download a pretrained model into the `weights/` folder.

We provide the following models (scores are with online default keyframes):

| `--config`  | Model  | Abs Diff↓| Sq Rel↓ | delta < 1.05↑| Chamfer↓ | F-Score↑ |
|-------------|----------|--------------------|---------|---------|--------------|----------|
| [`hero_model.yaml`](https://drive.google.com/file/d/1hCuKZjEq-AghrYAmFxJs_4eeixIlP488/view?usp=sharing) | Metadata + Resnet Matching | 0.0868 | 0.0127 | 74.26 | 5.69 | 0.680 |
| [`dot_product_model.yaml`](https://drive.google.com/file/d/13lW-VPgsl2eAo95E87RKWoK8KUZelkUK/view?usp=sharing) | Dot Product + Resnet Matching | 0.0910 | 0.0134 | 71.90 | 5.92 | 0.667 |

`hero_model` is the main model.

## Run simple demo

Run the following (note: most of the arguments are ignored). This will ask you to select the folder with images; navigate to the folder and double click on it; then select "Open".
```
CUDA_VISIBLE_DEVICES=0 python test_simple.py --name HERO_MODEL --output_base_path OUTPUT_PATH --config_file configs/models/hero_model.yaml --load_weights_from_checkpoint weights/hero_model.ckpt --data_config configs/data/vdr_dense.yaml --num_workers 8 --batch_size 1 --fast_cost_volume --run_fusion --depth_fuser open3d --fuse_color \
```
This will save a depth image (in **millimeters**) in the simplerecon root folder. 

If you want to visualize the saved depth image, you can run the following (e.g., with ipython):
```
import numpy as np
import matplotlib.pyplot as plt
depth = np.load("./depth.millimeters.frame-000007.npy")
plt.imshow(depth); plt.show()
```






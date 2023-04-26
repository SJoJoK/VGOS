# VGOS

Official code for the IJCAI 2023 paper "VGOS: Voxel Grid Optimization for View Synthesis from Sparse Inputs".

## Installation

```
git clone git@github.com:SJoJoK/VGOS.git
cd VGOS
pip install -r requirements.txt
```

[Pytorch](https://pytorch.org/) and [torch_scatter](https://github.com/rusty1s/pytorch_scatter) installation is machine dependent, please install the correct version for your machine. 

We conduct our experiments with torch 1.13.0+cu117 and torch-scatter 2.1.0+pt113cu117 on a single NVIDIA GeForce RTX 3090 GPU (24 GB).


## Directory structure for the datasets

<details>
  <summary> (click to expand) </summary>
    
    data
    ├── nerf_synthetic     # Link: https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
    │   └── [chair|drums|ficus|hotdog|lego|materials|mic|ship]
    │       ├── [train|val|test]
    │       │   └── r_*.png
    │       └── transforms_[train|val|test].json
    │
    └── nerf_llff_data     # Link: https://drive.google.com/drive/folders/14boI-o5hGO9srnWaaogTU5_ji7wkX2S7
        └── [fern|flower|fortress|horns|leaves|orchids|room|trex]
            ├── poses_bounds.npy
            └── [images_2|images_4]

</details>

## GO

- Training

  The `PSNR`, `SSIM` and `LPIPS` of the testset will be evaluated after training.

  ```zsh
  $ python run.py --config configs/llff_3v/room.py --render_test_get_metric
  ```

- Render w/o re-training

  ```zsh
  $ python run.py --config configs/llff_3v/room.py --render_only --render_video
  ```
- Re-conduct the experiments in the paper

  ```zsh
  $ zsh train_script/llff/llff_3v.sh
  $ zsh train_script/blender/blender_4v.sh
  ```

  Feel free to modify the train scripts to suit your own needs, such as adding experiments under different training views or tuning parameters (please refer to the comments in configs/default.py for more information about the parameters).
  
  Please note that the random selection of training views may result in fluctuations in performance. Therefore, the metrics presented for comparisons in our paper are the average scores of five experiments with different viewpoint samples. It is common to observe results that are slightly different, either higher or lower, than those presented in the paper.
## Acknowledgement

The code base is heavily based on [DVGOv2](https://github.com/sunset1995/DirectVoxGO) implementation, and the computation of depth smoothness loss is modified from [RegNeRF](https://github.com/google-research/google-research/tree/master/regnerf). Thanks for sharing!
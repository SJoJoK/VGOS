import os, sys, copy, glob, json, time, random, argparse
from shutil import copyfile
from tqdm import tqdm, trange

import mmcv
import imageio
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import kornia.losses
from lib import utils, dvgo, dmpigo
from lib.load_data import load_data
from torch_efficient_distloss import flatten_eff_distloss
from torch.utils.tensorboard import SummaryWriter
from open3d.visualization.tensorboard_plugin import summary
# Utility function to convert Open3D geometry to a dictionary format
from open3d.visualization.tensorboard_plugin.util import to_dict_batch
import matplotlib.pyplot as plt

@torch.no_grad()
def show_voxel_plt(model,thres,stage,step):
    denseGrid = model.density.get_dense_grid()[0] #c*x*y*z
    alpha = model.activate_density(denseGrid)
    mask = (alpha > thres).squeeze(dim=0).cpu().numpy()
    colors=alpha.repeat(3,1,1,1).permute(1,2,3,0).cpu().numpy()
    
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(mask,
            facecolors=colors,
            edgecolors='k',
            linewidth=0.5)
    ax.set_aspect('equal')
    # buf = io.BytesIO()
    plt.savefig(f'tmp/{stage}_train/voxel_{step}')
    
@torch.no_grad()
def log_voxel_o3d(model,thres,stage,step,writer,metrics,fine):
    if stage == "coarse":
        if cfg.data.ndc:
            alpha = model.activate_density(model.density.get_dense_grid()+model.act_shift.get_dense_grid()).squeeze().cpu().numpy()
        else:
            alpha = model.activate_density(model.density.get_dense_grid()).squeeze().cpu().numpy()
        rgb = torch.sigmoid(model.k0.get_dense_grid()).squeeze().permute(1,2,3,0).cpu().numpy()
        xyz = np.stack((alpha > thres).nonzero(), -1)
        N_down = 0
        while (not len(xyz)) or (N_down <3):
            thres/=2
            N_down+=1
            xyz = np.stack((alpha > thres).nonzero(), -1)
        if len(xyz):
            color = rgb[xyz[:,0], xyz[:,1], xyz[:,2]]
            xyz_min = np.array([0,0,0])
            xyz_max = np.array(alpha.shape)
            points = xyz / alpha.shape * (xyz_max - xyz_min) + xyz_min # N*3
            point_colors = color[:, :3] # N*3
            # if len(points) > 300000:
            #     points=points[np.random.choice(points.shape[0], 300000, replace=False), :]
            #     point_colors=point_colors[np.random.choice(point_colors.shape[0], 300000, replace=False), :]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(point_colors)
            writer.add_3d(f'{stage}/voxelGrid', to_dict_batch([pcd]), step=step)
            metrics.update({
                f'{stage}/voxelGrid': wandb.Object3D(np.append(points,utils.to8b(point_colors),axis=1))
            })   
    elif stage == "fine" and fine:
        log_voxel_size=[64,64,32]
        density_grid = model.density.get_dense_grid()
        k0_grid = model.k0.get_dense_grid()
        act_shift_grid = model.act_shift.get_dense_grid()
        density_grid_sub = F.interpolate(density_grid, log_voxel_size, align_corners=True, mode='trilinear')
        k0_grid_sub = F.interpolate(k0_grid, log_voxel_size, align_corners=True, mode='trilinear')
        act_shift_grid_sub = F.interpolate(act_shift_grid, log_voxel_size, align_corners=True, mode='trilinear')
        if cfg.data.ndc:
            alpha = model.activate_density(density_grid_sub+act_shift_grid_sub).squeeze().cpu().numpy()
        else:
            alpha = model.activate_density(density_grid_sub).squeeze().cpu().numpy()
        rgb = torch.sigmoid(k0_grid_sub).squeeze().permute(1,2,3,0).cpu().numpy()
        xyz = np.stack((alpha > thres).nonzero(), -1)
        N_down = 0
        while (not len(xyz)) or (N_down <3):
            thres/=2
            N_down+=1
            xyz = np.stack((alpha > thres).nonzero(), -1)
        if len(xyz):
            color = rgb[xyz[:,0], xyz[:,1], xyz[:,2]]
            xyz_min = np.array([0,0,0])
            xyz_max = np.array(alpha.shape)
            points = xyz / alpha.shape * (xyz_max - xyz_min) + xyz_min # N*3
            point_colors = color[:, :3] # N*3
            # if len(points) > 300000:
            #     points=points[np.random.choice(points.shape[0], 300000, replace=False), :]
            #     point_colors=point_colors[np.random.choice(point_colors.shape[0], 300000, replace=False), :]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(point_colors)
            writer.add_3d(f'{stage}/voxelGrid', to_dict_batch([pcd]), step=step)
            metrics.update({
                f'{stage}/voxelGrid': wandb.Object3D(np.append(points,utils.to8b(point_colors),axis=1))
            })        

def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--export_bbox_and_cams_only", type=str, default='',
                        help='export scene bbox and camera poses for debugging and 3d visualization')
    parser.add_argument("--export_coarse_only", type=str, default='')

    # testing options

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_test_result", action='store_true')
    parser.add_argument("--render_test_get_metric", action='store_true')
    parser.add_argument("--render_video_after_train", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--render_video_with_normal", action="store_true")
    parser.add_argument("--render_video_flipy", action='store_true')
    parser.add_argument("--render_video_rot90", default=0, type=int)
    parser.add_argument("--render_video_factor", type=float, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--render_depth_black", action='store_true')
    parser.add_argument("--dump_images", action='store_true')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')

    # logging/saving options
    parser.add_argument("--no_log", action='store_true')
    parser.add_argument("--i_print",   type=int, default=1000,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_detail_before",   type=int, default=0,
                        help='before which step to stop print detialed metric (every 10 step)')
    parser.add_argument("--i_render",   type=int, default=1000,
                        help='frequency of save the render result')
    parser.add_argument("--i_detail_render_before",   type=int, default=0,
                        help='before which step to stop logging voxel (every 100 step)')
    parser.add_argument("--i_voxel", type=int, default=5000,
                        help='frequency of save the visualized voxel')
    parser.add_argument("--i_fine_voxel", action='store_true',
                        help='to visualize the voxel in fine stage')
    parser.add_argument("--i_detail_voxel_before",   type=int, default=0,
                        help='before which step to stop logging voxel (every 100 step)')
    parser.add_argument("--i_val",   type=int, default=2000,
                        help='frequency of validation')
    parser.add_argument("--i_random_val",   type=int, default=2000,
                        help='frequency of validation (randomly)')
    parser.add_argument("--i_save_val_img",   action='store_true',
                        help='whether to save the val img')
    parser.add_argument("--i_render_index",   type=int, default=0,
                        help='index (of the splited set) of the render pose while saving the render result')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--wandb_entity",   type=str, default=None)
    parser.add_argument("--wandb_project",   type=str, default='SVGO')
    # config override
    #TODO: a more decent way...
    parser.add_argument("--config_override", action='store_true')
    parser.add_argument("--max_train_views", type=int, default=None, help="Maximum number of training views")
    parser.add_argument("--inc_steps", type=int, default=None, help="Maximum steps of Incremental Voxel Training")
    parser.add_argument("--x_mid", type=float,default=None, help="Used to calculate $P_{min_init}$ and $P_{max_init}$ in Sec 3.2")
    parser.add_argument("--y_mid", type=float,default=None, help="Used to calculate $P_{min_init}$ and $P_{max_init}$ in Sec 3.2")
    parser.add_argument("--z_mid", type=float,default=None, help="Used to calculate $P_{min_init}$ and $P_{max_init}$ in Sec 3.2")
    parser.add_argument("--x_init_ratio", type=float,default=None, help="Used to calculate $P_{min_init}$ and $P_{max_init}$ in Sec 3.2")
    parser.add_argument("--y_init_ratio", type=float,default=None, help="Used to calculate $P_{min_init}$ and $P_{max_init}$ in Sec 3.2")
    parser.add_argument("--z_init_ratio", type=float,default=None, help="Used to calculate $P_{min_init}$ and $P_{max_init}$ in Sec 3.2")
    parser.add_argument("--voxel_inc", action='store_true', help="To use Incremental Voxel Training")
    parser.add_argument("--coarse_steps", type=int, default=None, help="Number of coarse steps")
    parser.add_argument("--fine_steps", type=int, default=None, help="Number of fine steps")
    parser.add_argument("--expname", type=str, default=None, help="Name of the experiment")
    parser.add_argument("--gen_expname", action='store_true', help="Generate experiment name")
    parser.add_argument("--N_rand_sample", type=int, default=None, help="Batch size (number of random rays from sample views per optimization step)")
    parser.add_argument("--patch_size", type=int, default=None, help="Patch size for $L_{DS}$ ($L_{DS}$ is proposed in RegNeRF)")
    parser.add_argument("--ray_sampler", type=str, default=None, help="Ray sampling strategies")
    parser.add_argument("--weight_entropy_last", type=float, default=None, help="Weight of background entropy loss")
    parser.add_argument("--coarse_weight_entropy_last", type=float, default=None, help="Coarse weight of background entropy loss")
    parser.add_argument("--weight_rgbper", type=float, default=None, help="Weight of per-point rgb loss")
    parser.add_argument("--weight_distortion", type=float, default=None, help="Weight of distortion loss")
    parser.add_argument("--weight_tv_k0", type=float, default=None, help="Weight of total variation loss of color/feature voxel grid")
    parser.add_argument("--weight_tv_density", type=float, default=None, help="Weight of total variation loss of density voxel grid")
    parser.add_argument("--weight_tv_depth", type=float, default=None, help="Weight of $L_{DS}$")
    parser.add_argument("--coarse_weight_tv_k0", type=float, default=None, help="Coarse weight of total variation loss of color/feature voxel grid")
    parser.add_argument("--coarse_weight_tv_density", type=float, default=None, help="Coarse weight of total variation loss of density voxel grid")
    parser.add_argument("--coarse_weight_tv_depth", type=float, default=None, help="Coarse weight of $L_{DS}$")
    parser.add_argument("--weight_normal", type=float, default=None, help="Weight of $L{normal}$, smoothing the normal of the rendered patch was tried, but it doesn't work as well as $L{DS}$")
    parser.add_argument("--normal_w_grad", action='store_true', help="Whether to backpropagate the gradient of $L_{normal}$ to the ray origin and ray direction")
    parser.add_argument("--tv_depth_ndc", action='store_true', help="Total variation depth in NDC")
    parser.add_argument("--weight_tv_depth_start", type=float, default=None)
    parser.add_argument("--weight_tv_depth_end", type=float, default=None)
    parser.add_argument("--weight_inverse_depth", type=float, default=None, help="Weight of inverse_depth_smoothness_loss, not used in VGOS experiments")
    parser.add_argument("--tv_depth_before", type=int, default=None)
    parser.add_argument("--tv_depth_after", type=int, default=None)
    parser.add_argument("--tv_depth_decay", action='store_true')
    parser.add_argument("--distortion_for_sample", action='store_true', help="Whether to use distortion loss for sample views (distortion loss is proposed in Mip-NeRF and DVGOv2 implements it, not used in VGOS experiments)")
    parser.add_argument("--pervoxel_lr", action='store_true', help="View-count-based learning rate")
    parser.add_argument("--thres_grow_steps", type=int, default=None)
    parser.add_argument("--thres_start", type=float, default=None)
    parser.add_argument("--thres_end", type=float, default=None)
    parser.add_argument("--weight_entropy_ray", type=float, default=None, help="Weight of $L_{entropy}$ ($L_{entropy}$ is proposed in InfoNeRF, implemented but not used in VGOS experiments)")
    parser.add_argument("--weight_color_aware_smooth", type=float, default=None, help="Weight of color-aware smoothness loss")
    parser.add_argument("--coarse_weight_color_aware_smooth", type=float, default=None, help="Coarse weight of color-aware smoothness loss")
    parser.add_argument("--entropy_ray_type", type=str, default=None, help="Element used as probability to compute entropy")
    parser.add_argument("--entropy_type", type=str, default=None, help="Entropy type of $L_{entropy}$")
    parser.add_argument("--basedir", type=str, default=None, help="Base directory for the experiment")
    parser.add_argument("--testskip", type=int, default=None, help="Test skip for validation")
    parser.add_argument("--savedir", type=str, default=None, help="Directory to save the experiment results")
    parser.add_argument("--hardcode_train_views", type=int, nargs="+", default=[], help="Hardcode training views")
    






    return parser

@torch.no_grad()
def render_viewpoints(model, render_poses, HW, Ks, ndc, render_kwargs,
                      gt_imgs=None, savedir=None, dump_images=False,
                      render_factor=0, render_video_flipy=False, render_video_rot90=0,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False):
    '''Render images for the given viewpoints
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW = (HW/render_factor).astype(int)
        Ks[:, :2, :3] /= render_factor

    rgbs = []
    depths = []
    bgmaps = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []

    for i, c2w in enumerate(tqdm(render_poses)):

        H, W = HW[i]
        K = Ks[i]
        c2w = torch.Tensor(c2w)
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'depth', 'alphainv_last']
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()
        bgmap = render_result['alphainv_last'].cpu().numpy()

        rgbs.append(rgb)
        depths.append(depth)
        bgmaps.append(bgmap)
        if i==0:
            print('Testing', rgb.shape)

        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            psnrs.append(p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))

    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')
        if eval_ssim: print('Testing ssim', np.mean(ssims), '(avg)')
        if eval_lpips_vgg: print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
        if eval_lpips_alex: print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')

    if render_video_flipy:
        for i in range(len(rgbs)):
            rgbs[i] = np.flip(rgbs[i], axis=0)
            depths[i] = np.flip(depths[i], axis=0)
            bgmaps[i] = np.flip(bgmaps[i], axis=0)

    if render_video_rot90 != 0:
        for i in range(len(rgbs)):
            rgbs[i] = np.rot90(rgbs[i], k=render_video_rot90, axes=(0,1))
            depths[i] = np.rot90(depths[i], k=render_video_rot90, axes=(0,1))
            bgmaps[i] = np.rot90(bgmaps[i], k=render_video_rot90, axes=(0,1))

    if savedir is not None and dump_images:
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.array(rgbs)
    depths = np.array(depths)
    bgmaps = np.array(bgmaps)

    return rgbs, depths, bgmaps

@torch.no_grad()
def render_viewpoints_and_metric(model, render_poses, HW, Ks, ndc, render_kwargs,
                                gt_imgs=None, savedir=None, dump_images=False,
                                render_factor=0, render_video_flipy=False, render_video_rot90=0,
                                eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False):
    '''Render images for the given viewpoints; return evaluation metrics if gt given;
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    rgbs = []
    depths = []
    bgmaps = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []
    for i, c2w in enumerate(tqdm(render_poses)):

        H, W = HW[i]
        K = Ks[i]
        c2w = torch.Tensor(c2w)
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'depth', 'alphainv_last']
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()
        bgmap = render_result['alphainv_last'].cpu().numpy()

        rgbs.append(rgb)
        depths.append(depth)
        bgmaps.append(bgmap)
        if i==0:
            print('Testing', rgb.shape)

        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            psnrs.append(p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))

    if render_video_flipy:
        for i in range(len(rgbs)):
            rgbs[i] = np.flip(rgbs[i], axis=0)
            depths[i] = np.flip(depths[i], axis=0)
            bgmaps[i] = np.flip(bgmaps[i], axis=0)

    if render_video_rot90 != 0:
        for i in range(len(rgbs)):
            rgbs[i] = np.rot90(rgbs[i], k=render_video_rot90, axes=(0,1))
            depths[i] = np.rot90(depths[i], k=render_video_rot90, axes=(0,1))
            bgmaps[i] = np.rot90(bgmaps[i], k=render_video_rot90, axes=(0,1))

    if savedir is not None and dump_images:
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.array(rgbs)
    depths = np.array(depths)
    bgmaps = np.array(bgmaps)

    return rgbs, depths, bgmaps, psnrs, ssims, lpips_alex, lpips_vgg

# @torch.no_grad()
def render_viewpoints_with_normal(model, render_poses, HW, Ks, ndc, render_kwargs,
                      gt_imgs=None, savedir=None, dump_images=False,
                      render_factor=0, render_video_flipy=False, render_video_rot90=0,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False):
    '''Render images for the given viewpoints;
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)
    model.requires_grad_(False)
    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW = (HW/render_factor).astype(int)
        Ks[:, :2, :3] /= render_factor

    rgbs = []
    depths = []
    bgmaps = []
    normals = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []

    for i, c2w in enumerate(tqdm(render_poses)):

        H, W = HW[i]
        K = Ks[i]
        c2w = torch.Tensor(c2w)
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'depth', 'alphainv_last', 'normal']
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, **{**render_kwargs,'render_normal':True}).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].detach().cpu().numpy()
        depth = render_result['depth'].detach().cpu().numpy()
        bgmap = render_result['alphainv_last'].detach().cpu().numpy()
        normal = render_result['normal'].detach().cpu().numpy()
        rgbs.append(rgb)
        depths.append(depth)
        bgmaps.append(bgmap)
        normals.append(normal)
        if i==0:
            print('Testing', rgb.shape)

        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            psnrs.append(p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))

    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')
        if eval_ssim: print('Testing ssim', np.mean(ssims), '(avg)')
        if eval_lpips_vgg: print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
        if eval_lpips_alex: print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')

    if render_video_flipy:
        for i in range(len(rgbs)):
            rgbs[i] = np.flip(rgbs[i], axis=0)
            depths[i] = np.flip(depths[i], axis=0)
            bgmaps[i] = np.flip(bgmaps[i], axis=0)
            normals[i] = np.flip(normals[i], axis=0)

    if render_video_rot90 != 0:
        for i in range(len(rgbs)):
            rgbs[i] = np.rot90(rgbs[i], k=render_video_rot90, axes=(0,1))
            depths[i] = np.rot90(depths[i], k=render_video_rot90, axes=(0,1))
            bgmaps[i] = np.rot90(bgmaps[i], k=render_video_rot90, axes=(0,1))
            normals[i] = np.rot90(normals[i], k=render_video_rot90, axes=(0,1))

    if savedir is not None and dump_images:
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.array(rgbs)
    depths = np.array(depths)
    bgmaps = np.array(bgmaps)
    normals = np.array(normals)
    model.requires_grad_(True)
    return rgbs, depths, bgmaps, normals

def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(777)
    np.random.seed(args.seed)
    random.seed(777)

def load_everything(args, cfg):
    '''Load images / poses / camera settings / data split.
    '''
    data_dict = load_data(cfg.data)

    # remove useless field
    kept_keys = {
            'hwf', 'HW', 'Ks', 'near', 'far', 'near_clip',
            'i_train', 'i_val', 'i_test', 'irregular_shape',
            'poses', 'render_poses', 'images'}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    return data_dict

def _compute_bbox_by_cam_frustrm_bounded(cfg, HW, Ks, poses, i_train, near, far):
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        if cfg.data.ndc:
            pts_nf = torch.stack([rays_o+rays_d*near, rays_o+rays_d*far])
        else:
            pts_nf = torch.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
    return xyz_min, xyz_max

def _compute_bbox_by_cam_frustrm_unbounded(cfg, HW, Ks, poses, i_train, near_clip):
    # Find a tightest cube that cover all camera centers
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        pts = rays_o + rays_d * near_clip
        xyz_min = torch.minimum(xyz_min, pts.amin((0,1)))
        xyz_max = torch.maximum(xyz_max, pts.amax((0,1)))
    center = (xyz_min + xyz_max) * 0.5
    radius = (center - xyz_min).max() * cfg.data.unbounded_inner_r
    xyz_min = center - radius
    xyz_max = center + radius
    return xyz_min, xyz_max

def compute_bbox_by_cam_frustrm(args, cfg, HW, Ks, poses, i_train, near, far, **kwargs):
    print('compute_bbox_by_cam_frustrm: start')
    if cfg.data.unbounded_inward:
        xyz_min, xyz_max = _compute_bbox_by_cam_frustrm_unbounded(
                cfg, HW, Ks, poses, i_train, kwargs.get('near_clip', None))

    else:
        xyz_min, xyz_max = _compute_bbox_by_cam_frustrm_bounded(
                cfg, HW, Ks, poses, i_train, near, far)
    print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
    print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)
    print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max

@torch.no_grad()
def compute_bbox_by_coarse_geo(model_class, model_path, thres):
    print('compute_bbox_by_coarse_geo: start')
    eps_time = time.time()
    model = utils.load_model(model_class, model_path)
    interp = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, model.world_size[0]),
        torch.linspace(0, 1, model.world_size[1]),
        torch.linspace(0, 1, model.world_size[2]),
    ), -1)
    dense_xyz = model.xyz_min * (1-interp) + model.xyz_max * interp
    density = model.density(dense_xyz)
    alpha = model.activate_density(density)
    mask = (alpha > thres)
    active_xyz = dense_xyz[mask]
    xyz_min = active_xyz.amin(0)
    xyz_max = active_xyz.amax(0)
    print('compute_bbox_by_coarse_geo: xyz_min', xyz_min)
    print('compute_bbox_by_coarse_geo: xyz_max', xyz_max)
    eps_time = time.time() - eps_time
    print('compute_bbox_by_coarse_geo: finish (eps time:', eps_time, 'secs)')
    return xyz_min, xyz_max

def create_new_model(cfg, cfg_model, cfg_train, xyz_min, xyz_max, stage, coarse_ckpt_path):
    model_kwargs = copy.deepcopy(cfg_model)
    num_voxels = model_kwargs.pop('num_voxels')
    if len(cfg_train.pg_scale):
        num_voxels = int(num_voxels / (2**len(cfg_train.pg_scale)))

    if cfg.data.ndc:
        print(f'scene_rep_reconstruction ({stage}): \033[96muse multiplane images\033[0m')
        model = dmpigo.DirectMPIGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            **model_kwargs)
    # elif cfg.data.unbounded_inward:
    #     print(f'scene_rep_reconstruction ({stage}): \033[96muse contraced voxel grid (covering unbounded)\033[0m')
    #     model = dcvgo.DirectContractedVoxGO(
    #         xyz_min=xyz_min, xyz_max=xyz_max,
    #         num_voxels=num_voxels,
    #         **model_kwargs)
    else:
        print(f'scene_rep_reconstruction ({stage}): \033[96muse dense voxel grid\033[0m')
        model = dvgo.DirectVoxGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            mask_cache_path=coarse_ckpt_path,
            **model_kwargs)
    model = model.to(device)
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
    return model, optimizer

def load_existed_model(args, cfg, cfg_train, reload_ckpt_path):
    if cfg.data.ndc:
        model_class = dmpigo.DirectMPIGO
    # elif cfg.data.unbounded_inward:
    #     model_class = dcvgo.DirectContractedVoxGO
    else:
        model_class = dvgo.DirectVoxGO
    model = utils.load_model(model_class, reload_ckpt_path).to(device)
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
    model, optimizer, start = utils.load_checkpoint(
            model, optimizer, reload_ckpt_path, args.no_reload_optimizer)
    return model, optimizer, start

def scene_rep_reconstruction(args, cfg, cfg_model, cfg_train, 
                             xyz_min, xyz_max, data_dict, stage, 
                             coarse_ckpt_path=None, writer=None):
    # init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if abs(cfg_model.world_bound_scale - 1) > 1e-9:
        xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift
    HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images = [
        data_dict[k] for k in [
            'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses', 'images'
        ]
    ]
    i_train_ori = i_train
    #Sparse Input
    if len(cfg.sparse_train.hardcode_train_views):
        print('Original training views:', i_train)
        i_train = np.array(cfg.sparse_train.hardcode_train_views)
        print('Hardcoded train views:', i_train)
        if not args.no_log:
            writer.add_images('sparse_train/inputs',images[i_train],dataformats='NHWC')
            wandb.log({'sparse_train/inputs':i_train},commit=False)
    elif cfg.sparse_train.max_train_views > 0:
        print('Original training views:', i_train)
        i_train = np.random.choice(i_train, size=cfg.sparse_train.max_train_views, replace=False)
        print('Subsampled train views:', i_train)
        if not args.no_log:
            writer.add_images('sparse_train/inputs',images[i_train],dataformats='NHWC')
            wandb.log({'sparse_train/inputs':i_train},commit=False)

    # find whether there is existing checkpoint path
    last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_last.tar')
    if args.no_reload:
        reload_ckpt_path = None
    elif args.ft_path:
        reload_ckpt_path = args.ft_path
    elif os.path.isfile(last_ckpt_path):
        reload_ckpt_path = last_ckpt_path
    else:
        reload_ckpt_path = None

    # init model and optimizer
    if reload_ckpt_path is None:
        print(f'scene_rep_reconstruction ({stage}): train from scratch')
        model, optimizer = create_new_model(cfg, cfg_model, cfg_train, xyz_min, xyz_max, stage, coarse_ckpt_path)
        start = 0
        if cfg_model.maskout_near_cam_vox:
            model.maskout_near_cam_vox(poses[i_train,:3,3], near)
    else:
        print(f'scene_rep_reconstruction ({stage}): reload from {reload_ckpt_path}')
        model, optimizer, start = load_existed_model(args, cfg, cfg_train, reload_ckpt_path)

    # init rendering setup
    render_kwargs = {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'rand_bkgd': cfg.data.rand_bkgd,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
    }
    
    # init batch rays sampler
    def gather_training_rays():
        if data_dict['irregular_shape']:
            rgb_tr_ori = [images[i].to('cpu' if cfg.data.load2gpu_on_the_fly else device) for i in i_train]
        else:
            rgb_tr_ori = images[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)

        if cfg_train.ray_sampler == 'in_maskcache':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_in_maskcache_sampling(
                    rgb_tr_ori=rgb_tr_ori,
                    train_poses=poses[i_train],
                    HW=HW[i_train], Ks=Ks[i_train],
                    ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                    model=model, render_kwargs=render_kwargs)
        elif cfg_train.ray_sampler == 'flatten':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_flatten(
                rgb_tr_ori=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        else:
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays(
                rgb_tr=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        index_generator = dvgo.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
        batch_index_sampler = lambda: next(index_generator)
        return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler

    rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler = gather_training_rays()

    # Generate new rays for random poses
    def gather_random_rays():
        rays_o_rd, rays_d_rd, viewdirs_rd, imsz_rd = dvgo.get_random_rays(
            train_poses=poses[i_train_ori],
            HW=HW[i_train_ori], Ks=Ks[i_train_ori], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
            flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,n_poses=20)
        return rays_o_rd, rays_d_rd, viewdirs_rd, imsz_rd
        
    if cfg_train.N_rand_sample:  
        rays_o_rd_tr, rays_d_rd_tr, viewdirs_rd_tr, imsz_rd_tr = gather_random_rays()
    
    # view-count-based learning rate
    if cfg_train.pervoxel_lr:
        def per_voxel_init():
            if cfg_train.pervoxel_lr_for_rd:
                cnt = model.voxel_count_views(
                        rays_o_tr=rays_o_rd_tr, rays_d_tr= rays_d_rd_tr, imsz=imsz_rd_tr, near=near, far=far,
                        stepsize=cfg_model.stepsize, downrate=cfg_train.pervoxel_lr_downrate,
                        irregular_shape=data_dict['irregular_shape'])
                optimizer.set_pervoxel_lr(cnt.float() / cnt.max())
                model.mask_cache.mask[cnt.squeeze() <= 2] = False
            else:
                cnt = model.voxel_count_views(
                        rays_o_tr=rays_o_tr, rays_d_tr=rays_d_tr, imsz=imsz, near=near, far=far,
                        stepsize=cfg_model.stepsize, downrate=cfg_train.pervoxel_lr_downrate,
                        irregular_shape=data_dict['irregular_shape'])
                optimizer.set_pervoxel_lr(cnt.float() / cnt.max())
                model.mask_cache.mask[cnt.squeeze() <= 2] = False
        per_voxel_init()

    if cfg_train.maskout_lt_nviews > 0:
        model.update_occupancy_cache_lt_nviews(
                rays_o_tr, rays_d_tr, imsz, render_kwargs, cfg_train.maskout_lt_nviews)
    
    # Init Increment (between 0 and 1)
    x_mid = cfg_train.x_mid
    y_mid = cfg_train.y_mid
    z_mid = cfg_train.z_mid
    voxel_inc_lower_init = torch.tensor([
        x_mid-cfg_train.x_init_ratio*(x_mid),
        y_mid-cfg_train.y_init_ratio*(y_mid),
        z_mid-cfg_train.z_init_ratio*(z_mid)
    ])
    voxel_inc_upper_init = torch.tensor([
        x_mid+cfg_train.x_init_ratio*(1-x_mid),
        y_mid+cfg_train.y_init_ratio*(1-y_mid),
        z_mid+cfg_train.z_init_ratio*(1-z_mid)
    ])
    
    # GOGO
    torch.cuda.empty_cache()
    psnr_lst = []
    time0 = time.time()
    global_step = -1
    for global_step in trange(1+start, 1+cfg_train.N_iters):
        metrics={}
        # renew occupancy grid
        if model.mask_cache is not None and (global_step + 500) % 1000 == 0:
            model.update_occupancy_cache()

        # progress scaling checkpoint
        if global_step in cfg_train.pg_scale:
            n_rest_scales = len(cfg_train.pg_scale)-cfg_train.pg_scale.index(global_step)-1
            cur_voxels = int(cfg_model.num_voxels / (2**n_rest_scales))
            if isinstance(model, dvgo.DirectVoxGO):
                model.scale_volume_grid(cur_voxels)
            elif isinstance(model, dmpigo.DirectMPIGO):
                model.scale_volume_grid(cur_voxels, model.mpi_depth)
            # elif isinstance(model, dcvgo.DirectContractedVoxGO):
            #     model.scale_volume_grid(cur_voxels)
            else:
                raise NotImplementedError
            optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
            model.act_shift -= cfg_train.decay_after_scale
            torch.cuda.empty_cache()

        # random sample rays
        if cfg_train.ray_sampler in ['flatten', 'in_maskcache']:
            sel_i = batch_index_sampler()
            target = rgb_tr[sel_i] # N*C, target color of the select ray
            rays_o = rays_o_tr[sel_i] # N*3, origin of the select ray
            rays_d = rays_d_tr[sel_i] # N*3, direction of the select ray
            viewdirs = viewdirs_tr[sel_i] # N*3, viewdir of the select ray
        elif cfg_train.ray_sampler == 'random':
            sel_b = torch.randint(rgb_tr.shape[0], [cfg_train.N_rand]) # batch index of the select ray
            sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand]) # H index of the select ray
            sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand]) # W index of the select ray
            target = rgb_tr[sel_b, sel_r, sel_c] # N*C, target color of the select ray
            rays_o = rays_o_tr[sel_b, sel_r, sel_c] # N*3, origin of the select ray
            rays_d = rays_d_tr[sel_b, sel_r, sel_c] # N*3, direction of the select ray
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c] # N*3, viewdir of the select ray
        else:
            raise NotImplementedError

        # random sample patches from random poses
        if cfg_train.N_rand_sample:
            if cfg_train.ray_sampler == 'random':
                n_patches = cfg_train.N_rand_sample // (cfg_train.patch_size ** 2)
                H, W = HW[0]
                # Sample pose(s)
                idx_img = np.random.randint(0, len(rays_o_rd_tr), size=(n_patches, 1))
                #TODO: Sample patch from one image
                # Sample start locations
                x0 = np.random.randint(0, W - cfg_train.patch_size + 1, size=(n_patches, 1, 1))
                y0 = np.random.randint(0, H - cfg_train.patch_size + 1, size=(n_patches, 1, 1))
                xy0 = np.concatenate([x0, y0], axis=-1)
                patch_idx = xy0 + np.stack(
                    np.meshgrid(np.arange(cfg_train.patch_size), np.arange(cfg_train.patch_size), indexing='xy'),
                    axis=-1).reshape(1, -1, 2)
                
                rays_o_rd_patch = rays_o_rd_tr[idx_img, patch_idx[Ellipsis, 1], patch_idx[Ellipsis, 0]].reshape(-1, 3)
                rays_d_rd_patch = rays_d_rd_tr[idx_img, patch_idx[Ellipsis, 1], patch_idx[Ellipsis, 0]].reshape(-1, 3)
                viewdirs_rd_patch = viewdirs_rd_tr[idx_img, patch_idx[Ellipsis, 1], patch_idx[Ellipsis, 0]].reshape(-1, 3)
                
                sel_b_rd = torch.randint(rgb_tr.shape[0], [cfg_train.N_rand_sample]) # batch index of the select ray
                sel_r_rd = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand_sample]) # H index of the select ray
                sel_c_rd = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand_sample]) # W index of the select ray
                
                rays_o_rd = rays_o_rd_tr[sel_b_rd, sel_r_rd, sel_c_rd]
                rays_d_rd = rays_d_rd_tr[sel_b_rd, sel_r_rd, sel_c_rd]
                viewdirs_rd = viewdirs_rd_tr[sel_b_rd, sel_r_rd, sel_c_rd]
                
            elif cfg_train.ray_sampler != 'random':
                raise NotImplementedError 
        
        if cfg.data.load2gpu_on_the_fly:
            target = target.to(device)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            viewdirs = viewdirs.to(device)
        
        # Voxel Increment
        
        if cfg_train.voxel_inc and cfg_train.inc_steps>0:
            if global_step<=cfg_train.inc_steps:
                weight = min(global_step * 1.0 / cfg_train.inc_steps, 1.0) 
                voxel_inc_lower = voxel_inc_lower_init - weight * voxel_inc_lower_init
                voxel_inc_upper = voxel_inc_upper_init + weight * (1-voxel_inc_upper_init)
                model.set_inc_mask(voxel_inc_lower,voxel_inc_upper)
            else:
                model.unset_inc_mask()
           
        # Thres Growing    
        if cfg_train.thres_grow_steps>0 and global_step<=cfg_train.thres_grow_steps:
            growing_thres = cfg_train.thres_start + (global_step/cfg_train.thres_grow_steps) * (cfg_train.thres_end-cfg_train.thres_start)
            model.fast_color_thres=growing_thres
            
        # volume rendering
        render_result = model(
            rays_o, rays_d, viewdirs,
            global_step=global_step, is_train=True,
            **render_kwargs)
        
        render_rd = cfg_train.distortion_for_sample or cfg_train.weight_entropy_ray>0
        render_rd_patch_depth = (global_step<cfg_train.tv_depth_before and global_step>cfg_train.tv_depth_after and cfg_train.weight_tv_depth >0) or cfg_train.weight_inverse_depth>0
        render_rd_patch_normal = cfg_train.weight_normal>0 and not cfg_train.normal_w_grad
        render_rd_patch_normal_w_grad = cfg_train.weight_normal>0 and cfg_train.normal_w_grad
        render_rd_patch =  render_rd_patch_depth or render_rd_patch_normal or render_rd_patch_normal_w_grad
        if render_rd:
            render_rd_result = model(
                rays_o_rd, rays_d_rd, viewdirs_rd,
                global_step=global_step, is_train=True,
                **render_kwargs
                )
        
        if render_rd_patch:
            render_rd_patch_result = model(
                rays_o_rd_patch, rays_d_rd_patch, viewdirs_rd_patch,
                global_step=global_step, is_train=True,
                **{**render_kwargs,
                   'render_depth':render_rd_patch_depth,
                   'render_normal':render_rd_patch_normal,
                   'render_normal_w_grad':render_rd_patch_normal_w_grad}
                )
        
        # gradient descent step
        optimizer.zero_grad(set_to_none=True)
        loss = 0
        mse_loss = F.mse_loss(render_result['rgb_marched'], target)
        loss += cfg_train.weight_main * mse_loss
        psnr = utils.mse2psnr(loss.detach())
        
        if cfg_train.weight_entropy_last > 0:
            pout = render_result['alphainv_last'].clamp(1e-6, 1-1e-6)
            entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
            loss += cfg_train.weight_entropy_last * entropy_last_loss

        if cfg_train.weight_entropy_ray > 0:
            entropy_ray_loss = utils.segment_entropy(src=render_result[cfg_train.entropy_ray_type],
                                                     index=render_result['ray_id'],
                                                     alphainv_last=render_result['alphainv_last'],
                                                     entropy_type=cfg_train.entropy_type)
            entropy_ray_loss = entropy_ray_loss.mean()
            loss += cfg_train.weight_entropy_ray * entropy_ray_loss
        
        if cfg_train.weight_nearclip > 0:
            near_thres = data_dict['near_clip'] / model.scene_radius[0].item()
            near_mask = (render_result['t'] < near_thres)
            density = render_result['raw_density'][near_mask]
            if len(density):
                nearclip_loss = (density - density.detach()).sum()
                loss += cfg_train.weight_nearclip * nearclip_loss
        
        if cfg_train.weight_distortion > 0:
            if cfg_train.distortion_for_sample:
                n_max = render_rd_result['n_max']
                s = render_rd_result['s']
                w = render_rd_result['weights']
                ray_id = render_rd_result['ray_id']
            else:
                n_max = render_result['n_max']
                s = render_result['s']
                w = render_result['weights']
                ray_id = render_result['ray_id']
            loss_distortion = flatten_eff_distloss(w, s, 1/n_max, ray_id)
            loss += cfg_train.weight_distortion * loss_distortion
        
        if cfg_train.weight_rgbper > 0:
            rgbper = (render_result['raw_rgb'] - target[render_result['ray_id']]).pow(2).sum(-1)
            rgbper_loss = (rgbper * render_result['weights'].detach()).sum() / len(rays_o)
            loss += cfg_train.weight_rgbper * rgbper_loss
        
        if global_step<cfg_train.tv_depth_before and global_step>cfg_train.tv_depth_after and cfg_train.weight_tv_depth >0:
            depth = render_rd_patch_result['depth'].reshape(-1,cfg_train.patch_size,cfg_train.patch_size,1)
            depth_tvnorm_loss=utils.compute_tv_norm(depth).mean()
            if cfg_train.tv_depth_decay:
                loss+=utils.compute_tvnorm_weight(global_step,cfg_train.tv_depth_before,cfg_train.weight_tv_depth_start, cfg_train.weight_tv_depth_end)*depth_tvnorm_loss
            else :
                loss+=cfg_train.weight_tv_depth*depth_tvnorm_loss
                
        if cfg_train.weight_inverse_depth >0:
            idepth = render_rd_patch_result['depth'].reshape(-1,1,cfg_train.patch_size,cfg_train.patch_size)
            irgb = render_rd_patch_result['rgb_marched'].reshape(-1,3,cfg_train.patch_size,cfg_train.patch_size)
            inverse_depth_loss = kornia.losses.inverse_depth_smoothness_loss(idepth,irgb).mean()
            loss+=cfg_train.weight_inverse_depth * inverse_depth_loss
            
            
        if cfg_train.weight_normal > 0:
            normal_patch = render_rd_patch_result['normal']
            normal_mean = F.normalize(normal_patch.mean(dim=0),dim=0)
            normal_loss = -(normal_patch@normal_mean).mean()
            loss+=cfg_train.weight_normal * normal_loss
                
        loss.backward()

        if global_step<cfg_train.tv_before and global_step>cfg_train.tv_after and global_step%cfg_train.tv_every==0:
            if cfg_train.weight_tv_density>0:
                model.density_total_variation_add_grad(
                    cfg_train.weight_tv_density/len(rays_o), global_step<cfg_train.tv_dense_before)
            if cfg_train.weight_tv_k0>0:
                model.k0_total_variation_add_grad(
                    cfg_train.weight_tv_k0/len(rays_o), global_step<cfg_train.tv_dense_before)
            if cfg_train.weight_color_aware_smooth>0:
                model.color_aware_voxel_smooth_add_grad(
                    cfg_train.weight_color_aware_smooth/len(rays_o), global_step<cfg_train.tv_dense_before)

        optimizer.step()
        psnr_lst.append(psnr.item())

        # update lr
        decay_steps = cfg_train.lrate_decay * 1000
        decay_factor = 0.1 ** (1/decay_steps)
        for i_opt_g, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = param_group['lr'] * decay_factor

        # check log & save
        detailed = (global_step<args.i_detail_before and global_step%10==0)
        detailed_render = (global_step<args.i_detail_render_before and global_step%100==0)
        detailed_voxel = (global_step<args.i_detail_voxel_before and global_step%100==0)
        
        if (detailed or global_step%args.i_print==0) and not args.no_log:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            tqdm.write(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                       f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f} / '
                       f'Eps: {eps_time_str}')
            if cfg_train.weight_entropy_last > 0:
                writer.add_scalar(f'{stage}_train/weight_entropy_last', cfg_train.weight_entropy_last, global_step=global_step)
                writer.add_scalar(f'{stage}_train/entropy_last_loss', entropy_last_loss.item(), global_step=global_step)       
                metrics.update({
                f'{stage}_train/entropy_last_loss': entropy_last_loss.item()
            })  
            if cfg_train.weight_nearclip > 0:
                writer.add_scalar(f'{stage}_train/weight_nearclip', cfg_train.weight_nearclip, global_step=global_step)
                writer.add_scalar(f'{stage}_train/nearclip_loss', nearclip_loss.item(), global_step=global_step)
                metrics.update({
                f'{stage}_train/nearclip_loss': nearclip_loss.item()
            })  
            if cfg_train.weight_distortion > 0:
                writer.add_scalar(f'{stage}_train/weight_distortion', cfg_train.weight_distortion, global_step=global_step)
                writer.add_scalar(f'{stage}_train/distortion_loss', loss_distortion.item(), global_step=global_step)
                metrics.update({
                f'{stage}_train/distortion_loss': loss_distortion.item()
            })  
            if cfg_train.weight_rgbper > 0:
                writer.add_scalar(f'{stage}_train/weight_rgbper', cfg_train.weight_rgbper, global_step=global_step)     
                writer.add_scalar(f'{stage}_train/rgbper_loss', rgbper_loss.item(), global_step=global_step)
                metrics.update({
                f'{stage}_train/rgbper_loss': rgbper_loss.item()
            })
            if cfg_train.weight_tv_depth > 0 and global_step<cfg_train.tv_depth_before and global_step>cfg_train.tv_depth_after:     
                writer.add_scalar(f'{stage}_train/weight_tv_depth', cfg_train.weight_tv_depth, global_step=global_step)
                writer.add_scalar(f'{stage}_train/tv_depth_loss', depth_tvnorm_loss.item(), global_step=global_step)
                metrics.update({
                f'{stage}_train/tv_depth_loss': depth_tvnorm_loss.item()
            })
            if cfg_train.weight_entropy_ray > 0:
                writer.add_scalar(f'{stage}_train/weight_entropy_ray', cfg_train.weight_entropy_ray, global_step=global_step)       
                writer.add_scalar(f'{stage}_train/entropy_ray_loss', entropy_ray_loss.item(), global_step=global_step)       
                metrics.update({
                f'{stage}_train/entropy_last_loss': entropy_ray_loss.item()
            })
            if cfg_train.weight_normal > 0:
                writer.add_scalar(f'{stage}_train/weight_normal', cfg_train.weight_normal, global_step=global_step)       
                writer.add_scalar(f'{stage}_train/normal_loss', normal_loss.item(), global_step=global_step)       
                metrics.update({
                f'{stage}_train/normal_loss': normal_loss.item()
            }) 
            if cfg_train.weight_inverse_depth >0:
                writer.add_scalar(f'{stage}_train/weight_inverse_depth', cfg_train.weight_inverse_depth, global_step=global_step)       
                writer.add_scalar(f'{stage}_train/inverse_depth_loss', inverse_depth_loss.item(), global_step=global_step)       
                metrics.update({
                f'{stage}_train/inverse_depth_loss': inverse_depth_loss.item()
            }) 
            writer.add_scalar(f'{stage}_train/mse_loss',mse_loss.item(),global_step=global_step)
            writer.add_scalar(f'{stage}_train/loss', loss.item(), global_step=global_step)         
            writer.add_scalar(f'{stage}_train/psnr', np.mean(psnr_lst), global_step=global_step)
            metrics.update({
                f'{stage}_train/mse_loss':mse_loss.item(),
                f'{stage}_train/loss': loss.item(),
                f'{stage}_train/psnr': np.mean(psnr_lst),
            })  
            psnr_lst = []
        if (detailed_render or global_step%args.i_render==0) and not args.no_log:
            # with torch.no_grad():
                train_idx = i_train[args.i_render_index]
                test_idx = i_test[args.i_render_index]
                rgbs, depths, bgmaps, normals = render_viewpoints_with_normal(
                    render_poses=poses[[train_idx,test_idx]],
                    HW=HW[[train_idx,test_idx]],
                    Ks=Ks[[train_idx,test_idx]],
                    **{
                        'model': model,
                        'ndc': cfg.data.ndc,
                        'render_kwargs':{
                            **render_kwargs,
                            'render_depth': True,
                            'render_normal':True
                        }
                    }
                    )
                depths_vis = depths
                dmin, dmax = np.percentile(depths_vis[bgmaps < 0.1], q=[5, 95])
                depth_vis = plt.get_cmap('rainbow')(1 - np.clip((depths_vis - dmin) / (dmax - dmin), 0, 1)).squeeze()[..., :3]
                depth_vis[(bgmaps > 0.5).squeeze()]=[1,1,1]
                normal_vis = (normals*(bgmaps<0.1)+1)/2
                writer.add_image(f'{stage}_train/normal',normal_vis[0],global_step=global_step,dataformats='HWC')
                writer.add_image(f'{stage}_train/rgb',rgbs[0],global_step=global_step,dataformats='HWC')
                writer.add_image(f'{stage}_train/depth',depth_vis[0],global_step=global_step,dataformats='HWC')
                writer.add_image(f'{stage}_test/normal',normal_vis[1],global_step=global_step,dataformats='HWC')
                writer.add_image(f'{stage}_test/rgb',rgbs[1],global_step=global_step,dataformats='HWC')
                writer.add_image(f'{stage}_test/depth',depth_vis[1],global_step=global_step,dataformats='HWC')
                metrics.update({
                    f'{stage}_train/rgb': wandb.Image(utils.to8b(rgbs[0])),
                    f'{stage}_train/normal': wandb.Image(utils.to8b(normal_vis[0])),
                    f'{stage}_test/normal': wandb.Image(utils.to8b(normal_vis[1])),
                    f'{stage}_train/depth': wandb.Image(utils.to8b(depth_vis[0])),
                    f'{stage}_test/rgb':wandb.Image(utils.to8b(rgbs[1])),
                    f'{stage}_test/depth':wandb.Image(utils.to8b(depth_vis[1]))
                })
        if global_step%args.i_val==0 and not args.no_log:
            with torch.no_grad():
                rgbs, depths, bgmaps, psnrs, ssims, lpips_alex, lpips_vgg = render_viewpoints_and_metric(
                    render_poses=poses[i_val],
                    HW=HW[i_val],
                    Ks=Ks[i_val],
                    gt_imgs=[images[i].cpu().numpy() for i in i_val],
                    eval_ssim=True,
                    **{
                        'model': model,
                        'ndc': cfg.data.ndc,
                        'render_kwargs':{
                            **render_kwargs,
                            'render_depth': True
                        }
                    })
                if args.i_save_val_img:
                    depths_vis = depths
                    dmin, dmax = np.percentile(depths_vis[bgmaps < 0.1], q=[5, 95])
                    depth_vis = plt.get_cmap('rainbow')(1 - np.clip((depths_vis - dmin) / (dmax - dmin), 0, 1)).squeeze()[..., :3]
                    depth_vis[(bgmaps > 0.5).squeeze()]=[1,1,1]
                    writer.add_images(f'{stage}_val/rgb',rgbs,global_step=global_step,dataformats='NHWC')
                    writer.add_images(f'{stage}_val/depth',depth_vis,global_step=global_step,dataformats='NHWC')
                writer.add_scalar(f'{stage}_val/psnr', np.mean(psnrs), global_step=global_step)
                writer.add_scalar(f'{stage}_val/ssim', np.mean(ssims), global_step=global_step)
                metrics.update({
                    f'{stage}_val/psnr':np.mean(psnrs),
                    f'{stage}_val/ssim':np.mean(ssims),
                })
                
        if global_step%args.i_random_val==0 and not args.no_log:
            with torch.no_grad():
                i_random_val=np.array([0,1,2,3,24,25,14,61])
                rgbs, depths, bgmaps, psnrs, ssims, lpips_alex, lpips_vgg = render_viewpoints_and_metric(
                    render_poses=poses[i_random_val],
                    HW=HW[i_random_val],
                    Ks=Ks[i_random_val],
                    gt_imgs=[images[i].cpu().numpy() for i in i_random_val],
                    eval_ssim=True,
                    **{
                        'model': model,
                        'ndc': cfg.data.ndc,
                        'render_kwargs':{
                            **render_kwargs,
                            'render_depth': True
                        }
                    })
                depths_vis = depths
                dmin, dmax = np.percentile(depths_vis[bgmaps < 0.1], q=[5, 95])
                depth_vis = plt.get_cmap('rainbow')(1 - np.clip((depths_vis - dmin) / (dmax - dmin), 0, 1)).squeeze()[..., :3]
                depth_vis[(bgmaps > 0.5).squeeze()]=[1,1,1]
                writer.add_images(f'{stage}_val/rgb',rgbs,global_step=global_step,dataformats='NHWC')
                writer.add_images(f'{stage}_val/depth',depth_vis,global_step=global_step,dataformats='NHWC')
                writer.add_scalar(f'{stage}_val/psnr', np.mean(psnrs), global_step=global_step)
                writer.add_scalar(f'{stage}_val/ssim', np.mean(ssims), global_step=global_step)
                metrics.update({
                    f'{stage}_val/psnr':np.mean(psnrs),
                    f'{stage}_val/ssim':np.mean(ssims),
                })


        if (detailed_voxel or global_step%args.i_voxel==0) and not args.no_log:
            log_voxel_o3d(model,
                            cfg.fine_model_and_render.bbox_thres,
                            stage=stage,
                            step=global_step,
                            writer=writer,
                            metrics=metrics,
                            fine=args.i_fine_voxel)

       
        
        if global_step%args.i_weights==0:
            path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_{global_step:06d}.tar')
            torch.save({
                'global_step': global_step,
                'model_kwargs': model.get_kwargs(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', path)
        if metrics and not args.no_log:
            wandb.log(metrics,step=global_step,commit=(stage=='fine'))
    if global_step != -1:
        torch.save({
            'global_step': global_step,
            'model_kwargs': model.get_kwargs(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, last_ckpt_path)
        print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', last_ckpt_path)

def train(args, cfg, data_dict, writer):

    # init
    print('train: start')
    eps_time = time.time()
    os.makedirs(os.path.join(cfg.basedir, cfg.expname), exist_ok=True)
    with open(os.path.join(cfg.basedir, cfg.expname, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    cfg.dump(os.path.join(cfg.basedir, cfg.expname, 'config.py'))

    # coarse geometry searching (only works for inward bounded scenes)
    eps_coarse = time.time()
    xyz_min_coarse, xyz_max_coarse = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
    if cfg.coarse_train.N_iters > 0:
        scene_rep_reconstruction(
                args=args, cfg=cfg,
                cfg_model=cfg.coarse_model_and_render, cfg_train=cfg.coarse_train,
                xyz_min=xyz_min_coarse, xyz_max=xyz_max_coarse,
                data_dict=data_dict, stage='coarse', writer=writer)
        eps_coarse = time.time() - eps_coarse
        eps_time_str = f'{eps_coarse//3600:02.0f}:{eps_coarse//60%60:02.0f}:{eps_coarse%60:02.0f}'
        print('train: coarse geometry searching in', eps_time_str)
        coarse_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'coarse_last.tar')
    else:
        print('train: skip coarse geometry searching')
        coarse_ckpt_path = None

    # fine detail reconstruction
    eps_fine = time.time()
    if cfg.coarse_train.N_iters == 0:
        xyz_min_fine, xyz_max_fine = xyz_min_coarse.clone(), xyz_max_coarse.clone()
    else:
        xyz_min_fine, xyz_max_fine = compute_bbox_by_coarse_geo(
                model_class=dvgo.DirectVoxGO, model_path=coarse_ckpt_path,
                thres=cfg.fine_model_and_render.bbox_thres)
    scene_rep_reconstruction(
            args=args, cfg=cfg,
            cfg_model=cfg.fine_model_and_render, cfg_train=cfg.fine_train,
            xyz_min=xyz_min_fine, xyz_max=xyz_max_fine,
            data_dict=data_dict, stage='fine',
            coarse_ckpt_path=coarse_ckpt_path, writer=writer)
    eps_fine = time.time() - eps_fine
    eps_time_str = f'{eps_fine//3600:02.0f}:{eps_fine//60%60:02.0f}:{eps_fine%60:02.0f}'
    print('train: fine detail reconstruction in', eps_time_str)

    eps_time = time.time() - eps_time
    eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
    writer.add_scalar('result/time', eps_time)
    writer.add_text('result/time_str', eps_time_str)
    print('train: finish (eps time', eps_time_str, ')')
       
def cfg_override(args:argparse.Namespace,cfg:mmcv.Config):
    if args.max_train_views:
        cfg.sparse_train.max_train_views=args.max_train_views
    if args.inc_steps is not None:
        cfg.fine_train.inc_steps=args.inc_steps
    if args.fine_steps is not None:
        cfg.fine_train.N_iters=args.fine_steps
    if args.coarse_steps is not None:
        cfg.coarse_train.N_iters=args.coarse_steps
    if args.N_rand_sample is not None:
        cfg.coarse_train.N_rand_sample=args.N_rand_sample
        cfg.fine_train.N_rand_sample=args.N_rand_sample
    if args.patch_size is not None:
        cfg.fine_train.patch_size=args.patch_size
    if args.weight_tv_depth is not None:
        cfg.fine_train.weight_tv_depth=args.weight_tv_depth
    if args.ray_sampler is not None:
        cfg.fine_train.ray_sampler=args.ray_sampler
    if args.weight_tv_depth_start is not None:
        cfg.fine_train.weight_tv_depth_start=args.weight_tv_depth_start
    if args.weight_tv_depth_end is not None:
        cfg.fine_train.weight_tv_depth_end=args.weight_tv_depth_end
    if args.tv_depth_decay is not None:
        cfg.fine_train.tv_depth_decay=args.tv_depth_decay
    if args.tv_depth_before is not None:
        cfg.coarse_train.tv_depth_before=args.tv_depth_before
        cfg.fine_train.tv_depth_before=args.tv_depth_before
    if args.tv_depth_after is not None:
        cfg.coarse_train.tv_depth_after=args.tv_depth_after
        cfg.fine_train.tv_depth_after=args.tv_depth_after
    if args.weight_entropy_last is not None:
        cfg.fine_train.weight_entropy_last=args.weight_entropy_last
    if args.coarse_weight_entropy_last is not None:
        cfg.coarse_train.weight_entropy_last=args.coarse_weight_entropy_last
    if args.weight_rgbper is not None:
        cfg.fine_train.weight_rgbper=args.weight_rgbper
    if args.weight_distortion is not None:
        cfg.fine_train.weight_distortion=args.weight_distortion
    if args.weight_tv_k0 is not None:
        cfg.fine_train.weight_tv_k0=args.weight_tv_k0
    if args.weight_tv_density is not None:
        cfg.fine_train.weight_tv_density=args.weight_tv_density
    if args.distortion_for_sample is not None:
        cfg.fine_train.distortion_for_sample=args.distortion_for_sample
    if args.weight_entropy_ray is not None:
        cfg.fine_train.weight_entropy_ray=args.weight_entropy_ray
    if args.entropy_ray_type is not None:
        cfg.fine_train.entropy_ray_type=args.entropy_ray_type
    if args.weight_normal is not None:
        cfg.fine_train.weight_normal=args.weight_normal
    if args.normal_w_grad is not None:
        cfg.fine_train.normal_w_grad=args.normal_w_grad
    if args.entropy_type is not None:
        cfg.fine_train.entropy_type = args.entropy_type
    if args.pervoxel_lr is not None:
        cfg.fine_train.pervoxel_lr = args.pervoxel_lr  
    if args.x_mid is not None:
        cfg.fine_train.x_mid = args.x_mid
    if args.y_mid is not None:
        cfg.fine_train.y_mid = args.y_mid
    if args.z_mid is not None:
        cfg.fine_train.z_mid = args.z_mid
    if args.x_init_ratio is not None:
        cfg.fine_train.x_init_ratio = args.x_init_ratio
    if args.y_init_ratio is not None:
        cfg.fine_train.y_init_ratio = args.y_init_ratio
    if args.z_init_ratio is not None:
        cfg.fine_train.z_init_ratio = args.z_init_ratio
    if args.voxel_inc is not None:
        cfg.fine_train.voxel_inc = args.voxel_inc
    if args.basedir:
        cfg.basedir=args.basedir
    if args.expname:
        cfg.expname=args.expname
    if args.thres_grow_steps is not None:
        cfg.fine_train.thres_grow_steps = args.thres_grow_steps
    if args.thres_start is not None:
        cfg.fine_train.thres_start = args.thres_start
    if args.thres_end is not None:
        cfg.fine_train.thres_end = args.thres_end
    if args.weight_inverse_depth is not None:
        cfg.fine_train.weight_inverse_depth = args.weight_inverse_depth
    if args.weight_color_aware_smooth is not None:
        cfg.fine_train.weight_color_aware_smooth = args.weight_color_aware_smooth
    if args.coarse_weight_tv_depth is not None:
        cfg.coarse_train.weight_tv_depth=args.coarse_weight_tv_depth
    if args.coarse_weight_tv_density is not None:
        cfg.coarse_train.weight_tv_density=args.coarse_weight_tv_density
    if args.coarse_weight_tv_k0 is not None:
        cfg.coarse_train.weight_tv_k0=args.coarse_weight_tv_k0
    if args.coarse_weight_color_aware_smooth is not None:
        cfg.coarse_train.weight_color_aware_smooth=args.coarse_weight_color_aware_smooth
    if args.testskip is not None:
        cfg.data.testskip=args.testskip
    if args.hardcode_train_views is not []:
        cfg.sparse_train.hardcode_train_views=args.hardcode_train_views
    # if args.XXX is not None:
    #     cfg.fine_train.XXX = args.XXX
    return cfg

if __name__=='__main__':

    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    if args.config_override:
        cfg=cfg_override(args,cfg)
    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed_everything()

    # load images / poses / camera settings / data split
    data_dict = load_everything(args=args, cfg=cfg)

    # export scene bbox and camera poses in 3d for debugging and visualization
    if args.export_bbox_and_cams_only:
        print('Export bbox and cameras...')
        xyz_min, xyz_max = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
        poses, HW, Ks, i_train = data_dict['poses'], data_dict['HW'], data_dict['Ks'], data_dict['i_train']
        near, far = data_dict['near'], data_dict['far']
        if data_dict['near_clip'] is not None:
            near = data_dict['near_clip']
        cam_lst = []
        for c2w, (H, W), K in zip(poses[i_train], HW[i_train], Ks[i_train]):
            rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                    H, W, K, c2w, cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,)
            cam_o = rays_o[0,0].cpu().numpy()
            cam_d = rays_d[[0,0,-1,-1],[0,-1,0,-1]].cpu().numpy()
            cam_lst.append(np.array([cam_o, *(cam_o+cam_d*max(near, far*0.05))]))
        np.savez_compressed(args.export_bbox_and_cams_only,
            xyz_min=xyz_min.cpu().numpy(), xyz_max=xyz_max.cpu().numpy(),
            cam_lst=np.array(cam_lst))
        print('done')
        sys.exit()

    if args.export_coarse_only:
        print('Export coarse visualization...')
        with torch.no_grad():
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'coarse_last.tar')
            model = utils.load_model(dvgo.DirectVoxGO, ckpt_path).to(device)
            alpha = model.activate_density(model.density.get_dense_grid()).squeeze().cpu().numpy()
            rgb = torch.sigmoid(model.k0.get_dense_grid()).squeeze().permute(1,2,3,0).cpu().numpy()
        np.savez_compressed(args.export_coarse_only, alpha=alpha, rgb=rgb)
        print('done')
        sys.exit()

    if not args.no_log:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity)
        wandb.run.name = cfg.expname
        wandb.run.save()
        wandb.config.update(cfg._cfg_dict)
    writer = SummaryWriter(os.path.join(cfg.basedir, cfg.expname))
    
    # train
    if not args.render_only:
        train(args, cfg, data_dict,writer)

    # load model for rendring
    if args.render_test or args.render_train or args.render_video or args.render_test_get_metric or args.render_test_result or args.render_video_with_normal:
        if args.ft_path:
            ckpt_path = args.ft_path
        else:
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
        ckpt_name = ckpt_path.split('/')[-1][:-4]
        if cfg.data.ndc:
            model_class = dmpigo.DirectMPIGO
        # elif cfg.data.unbounded_inward:
        #     model_class = dcvgo.DirectContractedVoxGO
        else:
            model_class = dvgo.DirectVoxGO
        model = utils.load_model(model_class, ckpt_path).to(device)
        stepsize = cfg.fine_model_and_render.stepsize
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
                'render_depth': True,
            },
        }

    # render trainset and eval
    if args.render_train:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_train_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        rgbs, depths, bgmaps = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_train']],
                HW=data_dict['HW'][data_dict['i_train']],
                Ks=data_dict['Ks'][data_dict['i_train']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']],
                savedir=testsavedir, dump_images=args.dump_images,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(1 - depths / np.max(depths)), fps=30, quality=8)

    # render testset and eval
    if args.render_test or args.render_test_get_metric:
        if args.savedir is not None:
            testsavedir=os.path.join(args.savedir, cfg.expname)
        else:
            testsavedir=os.path.join(cfg.basedir, cfg.expname)
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        rgbs, depths, bgmaps, psnrs, ssims, lpips_alex, lpips_vgg = render_viewpoints_and_metric(
                render_poses=data_dict['poses'][data_dict['i_test']],
                HW=data_dict['HW'][data_dict['i_test']],
                Ks=data_dict['Ks'][data_dict['i_test']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
                savedir=testsavedir, dump_images=args.dump_images,
                eval_ssim=args.eval_ssim or args.render_test_get_metric,
                eval_lpips_alex=args.eval_lpips_alex or args.render_test_get_metric,
                eval_lpips_vgg=args.eval_lpips_vgg or args.render_test_get_metric,
                **render_viewpoints_kwargs)
        if args.render_test_get_metric and not args.no_log:
            writer.add_scalar('result/psnr', np.mean(psnrs))
            writer.add_scalar('result/ssim', np.mean(ssims))
            writer.add_scalar('result/lpips_alex', np.mean(lpips_alex))
            writer.add_scalar('result/lpipss_vgg', np.mean(lpips_vgg))
            wandb.log({
                'result/psnr':np.mean(psnrs),
                'result/ssim':np.mean(ssims),
                'result/lpips_alex':np.mean(lpips_alex),
                'result/lpips_vgg':np.mean(lpips_vgg)
                       })
        depths_vis = depths
        dmin, dmax = np.percentile(depths_vis[bgmaps < 0.1], q=[1, 99])
        depth_vis = plt.get_cmap('rainbow')(1 - np.clip((depths_vis - dmin) / (dmax - dmin), 0, 1)).squeeze()[..., :3]
        depth_vis[(bgmaps > 0.5).squeeze()]=[1,1,1]
        rgb8=utils.to8b(rgbs)
        depth8=utils.to8b(depth_vis)
        if args.render_depth_black:
            depth8=utils.to8b(depths / np.max(depths))
            depth8[bgmaps > 0.5]=0
        for i,img in enumerate(rgb8):
            imageio.imwrite(os.path.join(testsavedir, '{:03d}.png'.format(i)), img)
            imageio.imwrite(os.path.join(testsavedir, '{:03d}_depth.png'.format(i)), depth8[i])
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), rgb8, fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), depth8, fps=30, quality=8)
    
    # render video
    if args.render_video or args.render_video_after_train:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_video_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        rgbs, depths, bgmaps, normals = render_viewpoints_with_normal(
                render_poses=data_dict['render_poses'],
                HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                render_factor=args.render_video_factor,
                render_video_flipy=args.render_video_flipy,
                render_video_rot90=args.render_video_rot90,
                savedir=testsavedir, dump_images=args.dump_images,
                **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.normal.mp4'), utils.to8b((normals*(bgmaps<0.1)+1)/2), fps=30, quality=8)
        depths_vis = depths
        dmin, dmax = np.percentile(depths_vis[bgmaps < 0.1], q=[1, 99])
        depth_vis = plt.get_cmap('rainbow')(1 - np.clip((depths_vis - dmin) / (dmax - dmin), 0, 1)).squeeze()[..., :3]
        depth_vis[(bgmaps > 0.5).squeeze()]=[1,1,1]
        imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(depth_vis), fps=30, quality=8)
        if args.render_video_after_train and not args.no_log:
            wandb.log({
                'result/rgb':wandb.Video(os.path.join(testsavedir, 'video.rgb.mp4'),fps=30,format='mp4'),
                'result/depth':wandb.Video(os.path.join(testsavedir, 'video.depth.mp4'),fps=30,format='mp4'),
                       })       
    
    if not args.no_log:
        torch.cuda.empty_cache()
        wandb.finish()
    print('Done')


from copy import deepcopy

expname = None                    # experiment name
basedir = './logs/'               # where to store ckpts and logs

''' Template of data options
'''
data = dict(
    datadir=None,                 # path to dataset root folder
    dataset_type=None,            # blender | nsvf | blendedmvs | tankstemple | deepvoxels | co3d
    inverse_y=False,              # intrinsic mode (to support blendedmvs, nsvf, tankstemple)
    flip_x=False,                 # to support co3d
    flip_y=False,                 # to support co3d
    annot_path='',                # to support co3d
    split_path='',                # to support co3d
    sequence_name='',             # to support co3d
    load2gpu_on_the_fly=False,    # do not load all images into gpu (to save gpu memory)
    testskip=1,                   # subsample testset to preview results
    white_bkgd=False,             # use white background (note that some dataset don't provide alpha and with blended bg color)
    rand_bkgd=False,              # use random background during training
    half_res=False,               # [TODO]
    bd_factor=.75,
    movie_render_kwargs=dict(),

    # Below are forward-facing llff specific settings.
    ndc=False,                    # use ndc coordinate (only for forward-facing; not support yet)
    spherify=False,               # inward-facing
    factor=4,                     # [TODO]
    width=None,                   # enforce image width
    height=None,                  # enforce image height
    llffhold=8,                   # testsplit
    load_depths=False,            # load depth

    # Below are unbounded inward-facing specific settings.
    unbounded_inward=False,
    unbounded_inner_r=1.0,
)

''' Template of training options
'''
sparse_train = dict(
    max_train_views = -1,         # Number of sparse inputs, Negative means use all the train set
    hardcode_train_views = [],    # Index of sparse inputs
)

coarse_train = dict(
    N_iters=5000,                 # number of optimization steps
    N_rand=8192,                  # batch size (number of random rays per optimization step)
    N_rand_sample = 0,            # batch size (number of random rays from sample views per optimization step)
    lrate_density=1e-1,           # lr of density voxel grid
    lrate_k0=1e-1,                # lr of color/feature voxel grid
    lrate_rgbnet=1e-3,            # lr of the mlp to preduct view-dependent color
    lrate_decay=20,               # lr decay by 0.1 after every lrate_decay*1000 steps
    pervoxel_lr=True,             # view-count-based lr
    pervoxel_lr_for_rd=False,
    pervoxel_lr_downrate=1,       # downsampled image for computing view-count-based lr
    ray_sampler='random',         # ray sampling strategies
    weight_main=1.0,              # weight of photometric loss
    weight_entropy_last=0.01,     # weight of background entropy loss
    weight_nearclip=0,
    weight_distortion=0,
    weight_rgbper=0.1,            # weight of per-point rgb loss
    tv_every=1,                   # count total variation loss every tv_every step
    tv_after=0,                   # count total variation loss from tv_from step
    tv_before=100000,             # count total variation before the given number of iterations
    tv_dense_before=100000,       # count total variation densely before the given number of iterations
    weight_tv_density=0.0,        # weight of total variation loss of density voxel grid
    weight_tv_k0=0.0,             # weight of total variation loss of color/feature voxel grid
    weight_color_aware_smooth=0.0,
    pg_scale=[],                  # checkpoints for progressive scaling
    decay_after_scale=1.0,        # decay act_shift after scaling
    skip_zero_grad_fields=[],     # the variable name to skip optimizing parameters w/ zero grad in each iteration
    maskout_lt_nviews=0,
    voxel_inc=False,              # To use Incremental Voxel Training
    x_mid=0.5,                    # Used to calculate $P_{min_init}$ and $P_{max_init}$ in Sec 3.2
    y_mid=0.5,                    # Used to calculate $P_{min_init}$ and $P_{max_init}$ in Sec 3.2
    z_mid=0.5,                    # Used to calculate $P_{min_init}$ and $P_{max_init}$ in Sec 3.2
    x_init_ratio=1,               # Used to calculate $P_{min_init}$ and $P_{max_init}$ in Sec 3.2
    y_init_ratio=1,               # Used to calculate $P_{min_init}$ and $P_{max_init}$ in Sec 3.2
    z_init_ratio=1,               # Used to calculate $P_{min_init}$ and $P_{max_init}$ in Sec 3.2
    inc_steps=0,                  # Maximum steps of Incremental Voxel Training
    patch_size=8,                 # Patch size for $L_{DS}$ ($L_{DS}$ is proposed in RegNeRF)
    weight_tv_depth=0,            # Weight of $L_{DS}$
    tv_depth_decay=False,
    tv_depth_before=100000,
    tv_depth_after=0,
    weight_tv_depth_start=0,
    weight_tv_depth_end=0,
    distortion_for_sample=False,  # Whether to use distortion loss for sample views (distortion loss is proposed in Mip-NeRF and DVGOv2 implements it, not used in VGOS experiments)
    weight_entropy_ray=0,         # Weight of $L_{entropy}$ ($L_{entropy}$ is proposed in InfoNeRF, implemented but not used in VGOS experiments)
    weight_inverse_depth=0,       # Weight of inverse_depth_smoothness_loss, not used in VGOS experiments
    entropy_ray_type='weights',   # Element used as probability to compute entropy
    entropy_type='renyi',         # Entropy type of $L_{entropy}$
    normal_w_grad=False,          # Whether to backpropagate the gradient of $L_{normal}$ to the ray origin and ray direction
    weight_normal=0,              # Weight of $L_{normal}$, smoothing the normal of the rendered patch was tried, but it doesn't work as well as $L_{DS}$
    thres_grow_steps=0,
    thres_start=1e-4,
    thres_end=1e-3
    
    
)

fine_train = deepcopy(coarse_train)
fine_train.update(dict(
    N_iters=20000,
    pervoxel_lr=False,
    ray_sampler='in_maskcache',
    weight_entropy_last=0.001,
    weight_rgbper=0.01,
    pg_scale=[1000, 2000, 3000, 4000],
    skip_zero_grad_fields=['density', 'k0'],
))

''' Template of model and rendering options
'''
coarse_model_and_render = dict(
    num_voxels=1024000,           # expected number of voxel
    num_voxels_base=1024000,      # to rescale delta distance
    density_type='DenseGrid',     # DenseGrid, TensoRFGrid
    k0_type='DenseGrid',          # DenseGrid, TensoRFGrid
    density_config=dict(),
    k0_config=dict(),
    mpi_depth=128,                # the number of planes in Multiplane Image (work when ndc=True)
    nearest=False,                # nearest interpolation
    pre_act_density=False,        # pre-activated trilinear interpolation
    in_act_density=False,         # in-activated trilinear interpolation
    bbox_thres=1e-3,              # threshold to determine known free-space in the fine stage
    mask_cache_thres=1e-3,        # threshold to determine a tighten BBox in the fine stage
    rgbnet_dim=0,                 # feature voxel grid dim
    rgbnet_full_implicit=False,   # let the colors MLP ignore feature voxel grid
    rgbnet_direct=True,           # set to False to treat the first 3 dim of feature voxel grid as diffuse rgb
    rgbnet_depth=3,               # depth of the colors MLP (there are rgbnet_depth-1 intermediate features)
    rgbnet_width=128,             # width of the colors MLP
    alpha_init=1e-6,              # set the alpha values everywhere at the begin of training
    fast_color_thres=1e-7,        # threshold of alpha value to skip the fine stage sampled point
    maskout_near_cam_vox=True,    # maskout grid points that between cameras and their near planes
    world_bound_scale=1,          # rescale the BBox enclosing the scene
    stepsize=0.5,                 # sampling stepsize in volume rendering
)

fine_model_and_render = deepcopy(coarse_model_and_render)
fine_model_and_render.update(dict(
    num_voxels=160**3,
    num_voxels_base=160**3,
    rgbnet_dim=12,
    alpha_init=1e-2,
    fast_color_thres=1e-4,
    maskout_near_cam_vox=False,
    world_bound_scale=1.05,
))

del deepcopy

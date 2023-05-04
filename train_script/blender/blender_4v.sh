view=4
for cnt in 1
do
    for scene in "chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship"
    do
    python run.py \
    --config configs/nerf/${scene}.py --config_override --max_train_views $view --seed $RANDOM \
    --no_reload --no_reload_optimizer \
    --render_test_get_metric \
    --N_rand_sample 16384 --patch_size 8 --ray_sampler random \
    --inc_steps 256 \
    --x_init_ratio 0.8 --x_mid 0.5 \
    --y_init_ratio 0.8 --y_mid 0.5 \
    --z_init_ratio 0.8 --z_mid 0.5 --voxel_inc \
    --coarse_weight_tv_depth 0.0005 --weight_tv_depth 0.00001 \
    --coarse_weight_tv_density 0.0005 --coarse_weight_tv_k0 0.00005 --coarse_weight_color_aware_smooth 0.00005 \
    --weight_tv_density 0.00005 --weight_tv_k0 0.00001 --weight_color_aware_smooth 0.000005 \
    --i_render 100000 --i_val 100000 --i_voxel 100000 --i_random_val 100000 --i_print 100000 \
    --fine_steps 5000 --coarse_steps 5000 \
    --expname ${scene}_${view}v_${cnt} --basedir "./logs/blender_${view}v_${scene}";

    python run.py \
    --config configs/nerf/${scene}.py --config_override --max_train_views $view --seed $RANDOM \
    --no_reload --no_reload_optimizer \
    --render_test_get_metric \
    --i_render 100000 --i_val 100000 --i_voxel 100000 --i_random_val 100000 --i_print 100000 \
    --expname ${scene}_${view}v_base_${cnt} --basedir "./logs/blender_${view}v_${scene}";

################ RENDER #################
    
    # python run.py \
    # --config configs/nerf/${scene}.py --config_override --max_train_views $view \
    # --render_test --render_only --testskip 8 --render_depth_black \
    # --expname ${scene}_${view}v_hardcode --basedir "./logs/blender_${view}v_hardcode";

    # python run.py \
    # --config configs/nerf/${scene}.py --config_override --max_train_views $view \
    # --render_test --render_only --testskip 8 --render_depth_black \
    # --expname ${scene}_${view}v_hardcode_base --basedir "./logs/blender_${view}v_hardcode";

    done
done


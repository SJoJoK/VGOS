view=3
for cnt in 1
do
    for scene in "fortress" "horns" "room" "trex" "fern" "flower" "leaves" "orchids"
    do 
    python run.py \
    --no_reload --no_reload_optimizer \
    --render_test_get_metric \
    --config configs/llff/${scene}.py --config_override --max_train_views $view --seed $RANDOM \
    --inc_steps 256 \
    --x_init_ratio 1 --x_mid 0.5 \
    --y_init_ratio 1 --y_mid 0.5 \
    --z_init_ratio 0.005 --z_mid 1 --voxel_inc \
    --N_rand_sample 16384 --patch_size 8 --ray_sampler random \
    --weight_tv_depth 0.0005 --tv_depth_before 100000 --tv_depth_after 0 \
    --weight_distortion 0.0 \
    --weight_normal 0.0 \
    --i_render 100000 --i_val 100000 --i_voxel 100000 --i_random_val 100000 --i_print 100000 \
    --fine_steps 9000 \
    --entropy_type renyi --weight_entropy_ray 0.0 \
    --thres_grow_steps 0 --thres_start 0.001 --thres_end 0.001 \
    --weight_tv_density 0.00005 --weight_tv_k0 0.000005 --weight_color_aware_smooth 0.000005 \
    --expname "${scene}_${view}v_${cnt}" --basedir "./logs/llff_normal_${view}v_cpr";

    python run.py \
    --config configs/llff/${scene}.py --config_override --max_train_views $view --seed $RANDOM \
    --no_reload --no_reload_optimizer \
    --render_test_get_metric \
    --i_render 100000 --i_val 100000 --i_voxel 100000 --i_random_val 100000 --i_print 100000 \
    --expname "${scene}_${view}v_base_${cnt}" --basedir "./logs/llff_${view}v_cpr";

    ####### RENDER #######

    # python run.py \
    # --config configs/llff/${scene}.py --config_override --max_train_views $view \
    # --render_video --render_only \
    # --expname ${scene}_${view}v --basedir "./logs/llff_${view}v_cpr";

    # python run.py \
    # --config configs/llff/${scene}.py --config_override --max_train_views $view \
    # --render_video --render_only \
    # --expname ${scene}_${view}v_base --basedir "./logs/llff_${view}v_cpr";
    done
done


#!/bin/bash
cd "$(dirname "$0")"

SIM1_DIR=$(ls -td runs_sim_less_aggresive/asymmetric_update/trial_3_2*_seed_0 | grep -v continual | head -1)
echo "Using: $SIM1_DIR"

python3 sac_cleanrl.py \
    --render_mode rgb_array \
    --env_id SimEmbodiedAnt \
    --runs_directory runs_sim_less_aggresive/asymmetric_update \
    --exp_name trial_3_asym_continual_learning \
    --num_envs 1 \
    --weights_path $SIM1_DIR \
    --total_timesteps 120000 \
    --model_path ../../sim/assets/ant_with_camera_after_sys_id_real_less_aggresive.xml \
    --radius_back_and_forth 1.0 \
    --origin_back_and_forth 0.0 0.0 \
    --policy_frequency 20 \
    --policy_lr 1e-5 \
    --q_lr 3e-4 \
    --seed 0 \
    --cuda

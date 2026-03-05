#!/bin/bash

# Learn in simulation.
if [ "$1" == "sim" ]; then
    python3 sac_cleanrl.py \
        --render_mode rgb_array \
        --dt 0.12 \
        --env_id SimEmbodiedAnt \
        --learning_starts 2000 \
        --task_type back_and_forth \
        --model_path ../../sim/assets/ant_with_camera_after_sys_id.xml \
        --runs_directory runs_sim \
        --exp_name trial_1 \
        --use_layer_norm \
        --total_timesteps 30000
fi

# Learn on hardware.
if [ "$1" == "hw" ]; then
    python3 sac_cleanrl.py \
        --render_mode rgb_array \
        --dt 0.12 \
        --env_id HwEmbodiedAnt \
        --hw_config ../../embodied_ant_env/ant34.json \
        --learning_starts 2000 \
        --task_type back_and_forth \
        --runs_directory runs_hw \
        --exp_name trial_1 \
        --use_layer_norm \
        --seed 1 \
        # --eval True
fi

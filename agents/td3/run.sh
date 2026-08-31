#!/bin/bash

# Learn in simulation.
if [ "$1" == "sim" ]; then
    python3 td3_cleanrl.py \
        --render_mode rgb_array \
        --env_id SimEmbodiedAnt \
        --runs_directory runs_sim_test \
        --exp_name trial_1 \
        --num_envs 1
fi

# Learn on hardware.
if [ "$1" == "hw" ]; then
    python3 td3_cleanrl.py \
        --render_mode rgb_array \
        --dt 0.12 \
        --env_id HwEmbodiedAnt \
        --hw_config ../../embodied_ant_env/ant10.json \
        --learning_starts 2000 \
        --task_type back_and_forth \
        --runs_directory runs_timing \
        --exp_name trial_1 \
        --seed 1 \
        # --eval True
fi
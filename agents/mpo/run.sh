#!/bin/bash

# Learn in simulation.
if [ "$1" == "sim" ]; then
    python3 mpo_default.py \
        --render_mode rgb_array \
        --dt 0.15 \
        --env_id SimEmbodiedAnt \
        --runs_directory runs \
        --exp_name perf_check \
        --utd_ratio 1 \
fi

# Learn on hardware.
if [ "$1" == "hw" ]; then
    python3 mpo_default.py \
        --render_mode rgb_array \
        --dt 0.15 \
        --env_id HwEmbodiedAnt \
        --hw_config ../../embodied_ant_env/ant12.json \
        --learning_starts 500 \
        --task_type back_and_forth \
        --runs_directory runs_hw \
        --exp_name trial_1 \
        --seed 1 \
        # --eval True
fi

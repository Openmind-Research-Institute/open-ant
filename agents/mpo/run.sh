#!/bin/bash

# Learn in simulation.
if [ "$1" == "sim" ]; then
    python3 mpo_default.py \
        --render_mode rgb_array \
        --dt 0.15 \
        --env_id SimEmbodiedAnt \
        --runs_directory runs \
        --exp_name og_kl \
        --utd_ratio 3 \
        --ensemble 3 \
        --decouple_q_learning \
        --policy_learning_starts 1000 \
        --td_horizon 3
fi

# Learn on hardware.
if [ "$1" == "hw" ]; then
    python3 mpo_default.py \
        --render_mode rgb_array \
        --dt 0.15 \
        --env_id HwEmbodiedAnt \
        --hw_config ../../embodied_ant_env/ant12.json \
        --learning_starts 2000 \
        --task_type back_and_forth \
        --runs_directory runs_hw \
        --exp_name trial_3 \
        --seed 1 \
        --utd_ratio 3 \
        --ensemble 3 \
        --decouple_q_learning \
        --policy_learning_starts 1000 \
        --td_horizon 3
        # --eval True
fi

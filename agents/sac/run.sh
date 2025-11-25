#!/bin/bash

# learn in sim (single run, no Optuna)
if [ "$1" == "sim" ]; then
    python3 sac_cleanrl.py \
        --render_mode rgb_array \
        --dt 0.12 \
        --env_id SimEmbodiedAnt \
        --learning_starts 2000 \
        --task_type back_and_forth \
        --seed 1 \
        --total_timesteps 35000

    python3 sac_cleanrl.py \
        --render_mode rgb_array \
        --dt 0.12 \
        --env_id SimEmbodiedAnt \
        --learning_starts 2000 \
        --task_type back_and_forth \
        --seed 2

    python3 sac_cleanrl.py \
        --render_mode rgb_array \
        --dt 0.12 \
        --env_id SimEmbodiedAnt \
        --learning_starts 2000 \
        --task_type back_and_forth \
        --seed 3
    
    python3 sac_cleanrl.py \
        --render_mode rgb_array \
        --dt 0.12 \
        --env_id SimEmbodiedAnt \
        --learning_starts 2000 \
        --task_type back_and_forth \
        --seed 4

    python3 sac_cleanrl.py \
        --render_mode rgb_array \
        --dt 0.12 \
        --env_id SimEmbodiedAnt \
        --learning_starts 2000 \
        --task_type back_and_forth \
        --seed 5

    python3 sac_cleanrl.py \
        --render_mode rgb_array \
        --dt 0.12 \
        --env_id SimEmbodiedAnt \
        --learning_starts 2000 \
        --task_type back_and_forth \
        --seed 6

    python3 sac_cleanrl.py \
        --render_mode rgb_array \
        --dt 0.12 \
        --env_id SimEmbodiedAnt \
        --learning_starts 2000 \
        --task_type back_and_forth \
        --seed 7

    python3 sac_cleanrl.py \
        --render_mode rgb_array \
        --dt 0.12 \
        --env_id SimEmbodiedAnt \
        --learning_starts 2000 \
        --task_type back_and_forth \
        --seed 8

    python3 sac_cleanrl.py \
        --render_mode rgb_array \
        --dt 0.12 \
        --env_id SimEmbodiedAnt \
        --learning_starts 2000 \
        --task_type back_and_forth \
        --seed 9

    python3 sac_cleanrl.py \
        --render_mode rgb_array \
        --dt 0.12 \
        --env_id SimEmbodiedAnt \
        --learning_starts 2000 \
        --task_type back_and_forth \
        --seed 10

fi

# learn in sim with Optuna optimization
if [ "$1" == "sim_optuna" ]; then
    python3 experiments_optuna.py \
        --optuna_study_name "sac_ant_sim_study" \
        --optuna_n_trials 50 \
        --render_mode rgb_array \
        --dt 0.12 \
        --env_id SimEmbodiedAnt \
        --learning_starts 2000 \
        --optuna_storage "sqlite:///optuna_sac_ant_sim_study.db"
fi

# learn on hardware (single run, no Optuna)
if [ "$1" == "hw" ]; then
    python3 sac_cleanrl.py \
        --render_mode rgb_array \
        --dt 0.12 \
        --env_id HwEmbodiedAnt \
        --hw_config ../../embodied_ant_env/ant12.json \
        --learning_starts 2000 \
        --task_type forward \
        --render_mode human
fi


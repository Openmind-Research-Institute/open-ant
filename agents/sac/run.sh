#!/bin/bash

# learn in sim (single run, no Optuna)
if [ "$1" == "sim" ]; then
    python3 sac_cleanrl.py \
        --render_mode rgb_array \
        --dt 0.12 \
        --env_id SimEmbodiedAnt \
        --learning_starts 2000 \
        --capture_video true
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
    python3 sac_cleanrl_optuna.py \
        --render_mode rgb_array \
        --dt 0.12 \
        --env_id HwEmbodiedAnt \
        --hw_config ../../embodied_ant_env/ant12.json \
        --learning_starts 2000
fi


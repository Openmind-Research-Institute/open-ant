#!/bin/bash

# learn in sim (single run, no Optuna)
if [ "$1" == "sim_old" ]; then
    # Run several experiments in parallel using GNU parallel.
    # Requires the `parallel` command: sudo apt-get install parallel
    parallel -j 10 --halt soon,fail=1 --joblog parallel_sac.log '
        python3 sac_cleanrl_old.py \
            --render_mode rgb_array \
            --dt 0.12 \
            --env_id SimEmbodiedAnt \
            --learning_starts 2000 \
            --seed {1} \
            --batch_size 256 \
            --total_timesteps 30000 \
            --capture_video \
            > "run_seed_{1}.log" 2>&1
        # --capture_video \
    ' ::: 1 2 3 4 5 6 7 8 9 10
fi

if [ "$1" == "sim" ]; then
    python3 sac_cleanrl.py \
        --render_mode rgb_array \
        --dt 0.12 \
        --env_id SimEmbodiedAnt \
        --learning_starts 2000 \
        --seed 1 \
        --batch_size 256 \
        --task_type forward
        # --capture_video \
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


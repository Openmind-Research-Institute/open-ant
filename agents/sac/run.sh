#!/bin/bash

# Learn in simulation.
if [ "$1" == "sim" ]; then
    for gamma in $(seq 0.2 0.1 0.9); do
        for seed in $(seq 0 4); do
        python3 sac_cleanrl.py \
            --render_mode rgb_array \
            --env_id SimEmbodiedAnt \
            --runs_directory runs_analysis_gamma_multiple_seeds \
            --exp_name "gamma_${gamma}_seed_${seed}" \
            --num_envs 1 \
            --seed "$seed" \
            --gamma "$gamma" \
            --total_timesteps 100_000 &
        done
    done
    wait
fi

# Continual learning in simulation from previous gamma checkpoints.
if [ "$1" == "sim_continual" ]; then
    for gamma in $(seq 0.2 0.1 0.9); do
        for seed in $(seq 0 4); do
            WEIGHTS_PATH=$(ls -td "runs_analysis_gamma_multiple_seeds"/gamma_${gamma}_seed_${seed}_* 2>/dev/null | grep -v continual | head -1)
            if [ -z "$WEIGHTS_PATH" ]; then
                echo "ERROR: No prior run found for gamma=${gamma} seed=${seed} in runs_analysis_gamma_multiple_seeds"
                exit 1
            fi
            echo "gamma=${gamma} seed=${seed} loading weights from: ${WEIGHTS_PATH}"
            python3 sac_cleanrl.py \
                --render_mode rgb_array \
                --env_id SimEmbodiedAnt \
                --runs_directory runs_analysis_gamma_multiple_seeds \
                --exp_name "gamma_${gamma}_continual_seed_${seed}" \
                --num_envs 1 \
                --seed "$seed" \
                --gamma "$gamma" \
                --total_timesteps 300_000 \
                --model_path ../../sim/assets/ant_sim_2.xml \
                --weights_path "$WEIGHTS_PATH" &
        done
    done
    wait
fi

# Learn on hardware.
if [ "$1" == "hw" ]; then
    python3 sac_cleanrl.py \
        --render_mode rgb_array \
        --dt 0.12 \
        --env_id HwEmbodiedAnt \
        --hw_config ../../embodied_ant_env/ant12.json \
        --learning_starts 2000 \
        --task_type back_and_forth \
        --runs_directory runs_hw_new_refactored_code \
        --exp_name trial_1 \
        --seed 1 \
        # --eval True
fi
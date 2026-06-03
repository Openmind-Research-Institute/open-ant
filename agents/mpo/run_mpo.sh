#!/bin/bash

SEEDS=(0 1 4)

for SEED in "${SEEDS[@]}"; do
    echo "Running seed $SEED: sim1..."

    python3 mpo_default.py \
        --render_mode rgb_array \
        --env_id SimEmbodiedAnt \
        --runs_directory runs/60k_ensemble3_horizon3 \
        --exp_name trial_1_mpo \
        --total_timesteps 60000 \
        --dt 0.12 \
        --num_envs 1 \
        --radius_back_and_forth 1.0 \
        --origin_back_and_forth 0.0 0.0 \
        --utd_ratio 3 \
        --ensemble 3 \
        --decouple_q_learning \
        --policy_learning_starts 2000 \
        --td_horizon 3 \
        --seed $SEED \
        --cuda

    SIM1_DIR=$(ls -td runs/60k_ensemble3_horizon3/trial_1_mpo*_seed_${SEED} | grep -v continual | head -1)
    WEIGHTS_PATH="$SIM1_DIR/weights_and_args"

    echo "Sim1 run folder: $SIM1_DIR"
    echo "Running seed $SEED: continual learning..."

    python3 mpo_default.py \
        --render_mode rgb_array \
        --env_id SimEmbodiedAnt \
        --runs_directory runs/60k_ensemble3_horizon3 \
        --exp_name trial_1_mpo_continual_learning \
        --total_timesteps 120000 \
        --dt 0.12 \
        --num_envs 1 \
        --radius_back_and_forth 1.0 \
        --origin_back_and_forth 0.0 0.0 \
        --utd_ratio 3 \
        --ensemble 3 \
        --decouple_q_learning \
        --policy_learning_starts 2000 \
        --td_horizon 3 \
        --weights_path "$WEIGHTS_PATH" \
        --model_path ../../sim/assets/ant_with_camera_after_sys_id_real_less_aggresive.xml \
        --seed $SEED \
        --cuda

    echo "Done with seed $SEED"
done

echo "All seeds complete!"

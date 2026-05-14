#!/bin/bash

# Learn in simulation.
if [ "$1" == "sim" ]; then
    if [ "$2" == "multi_seed" ]; then
       SEEDS=(459 3956 1079 2205 4086 3849 4996 4998 3166 1798)
    else
        SEEDS=(1)
    fi
    for SEED in "${SEEDS[@]}"; do
        python3 mpo_default.py \
            --render_mode rgb_array \
            --total_timesteps 40000 \
            --dt 0.15 \
            --env_id SimEmbodiedAnt \
            --runs_directory runs \
            --exp_name retrace \
            --utd_ratio 3 \
            --ensemble 3 \
            --decouple_q_learning \
            --policy_learning_starts 1000 \
            --td_horizon 3 \
            --seed $SEED \
            # --weights_path /Users/mathieudecker/embodied-mujoco-ant/agents/mpo/runs/seed_459_trial_helios_20260507-180606_seed_459/weights_and_args \
            # --eval
        echo "trial ${SEED} done"
    done
fi

# Learn on hardware.
if [ "$1" == "hw" ]; then
    python3 mpo_default.py \
        --render_mode rgb_array \
        --dt 0.20 \
        --total_timesteps 60000 \
        --env_id HwEmbodiedAnt \
        --hw_config ../../embodied_ant_env/ant12.json \
        --learning_starts 2000 \
        --task_type back_and_forth \
        --runs_directory runs_hw \
        --exp_name larger_dt \
        --seed 1 \
        --utd_ratio 3 \
        --ensemble 3 \
        --decouple_q_learning \
        --policy_learning_starts 1000 \
        --td_horizon 3 \
        # --weights_path /Users/mathieudecker/embodied-mujoco-ant/agents/mpo/runs/seed_459_trial_helios_20260507-180606_seed_459/weights_and_args \
        # --eval
fi

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
            --total_timesteps 3000 \
            --dt 0.15 \
            --env_id SimEmbodiedAnt \
            --runs_directory runs \
            --exp_name "seed_${SEED}_trial" \
            --utd_ratio 3 \
            --ensemble 3 \
            --decouple_q_learning \
            --policy_learning_starts 1000 \
            --td_horizon 3 \
            --seed $SEED
            # --weights_path /Users/mathieudecker/embodied-mujoco-ant/agents/mpo/runs/new_baseline_20260506-085105_seed_1/weights_and_args
        echo "trial ${SEED} done"
    done
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
        --exp_name sim2real_cntd \
        --seed 1 \
        --utd_ratio 3 \
        --ensemble 3 \
        --decouple_q_learning \
        --policy_learning_starts 1000 \
        --td_horizon 3 \
        --weights_path /Users/mathieudecker/embodied-mujoco-ant/agents/mpo/runs_hw/sim2real_20260506-153614_seed_1/weights_and_args
        # --eval True
fi

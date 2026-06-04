#!/bin/bash
# Learn in simulation.
if [ "$1" == "sim" ]; then
        python3 mpo_default.py \
            --render_mode rgb_array \
            --total_timesteps 40000 \
            --dt 0.12 \
            --env_id SimEmbodiedAnt \
            --runs_directory runs_continous_learning \
            --exp_name retrace \
            --utd_ratio 3 \
            --ensemble 3 \
            --decouple_q_learning \
            --policy_learning_starts 2000 \
            --td_horizon 3 \
            --seed 0 \
            --cuda
fi


if [ "$1" == "sim_continual_learning" ]; then
    python3 mpo_default.py \
        --render_mode rgb_array \
        --env_id SimEmbodiedAnt \
        --total_timesteps 120_000 \
        --dt 0.12 \
        --runs_directory runs_continous_learning \
        --exp_name retrace_continual_learning \
        --utd_ratio 3 \
        --ensemble 3 \
        --decouple_q_learning \
        --policy_learning_starts 2000 \
        --td_horizon 3 \
        --weights_path ~/linc/open-ant/agents/mpo/runs_continous_learning/retrace_20260603-223153_seed_0/weights_and_args \
        --model_path ../../sim/assets/ant_with_camera_after_sys_id_real_less_aggresive.xml \
        --cuda \
        --seed 0
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

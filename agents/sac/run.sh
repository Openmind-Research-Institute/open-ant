#!/bin/bash

# Learn in simulation.
if [ "$1" == "sim" ]; then
    python3 sac_cleanrl.py \
        --render_mode rgb_array \
        --env_id SimEmbodiedAnt \
        --runs_directory runs_sim_less_aggresive \
        --exp_name trial_1 \
        --num_envs 1 \
        --radius_back_and_forth 1.0 \
        --origin_back_and_forth 0.0 0.0 \
        --cuda &
fi

if [ "$1" == "sim_continual_learning" ]; then
    python3 sac_cleanrl.py \
        --render_mode rgb_array \
        --env_id SimEmbodiedAnt \
        --runs_directory runs_sim_less_aggresive \
        --exp_name trial_1_continual_learning \
        --num_envs 1 \
        --weights_path /home/serena-liu/open-ant/agents/sac/runs_sim_less_aggresive/trial_1_20260529-011538_seed_42 \
        --offline_buffer_path /home/serena-liu/open-ant/agents/sac/runs_sim_less_aggresive/trial_1_20260529-011538_seed_42/replay_buffer \
        --total_timesteps 120_000 \
        --model_path ../../sim/assets/ant_with_camera_after_sys_id_real_less_aggresive.xml \
        --radius_back_and_forth 1.0 \
        --origin_back_and_forth 0.0 0.0 \
        --cuda &
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
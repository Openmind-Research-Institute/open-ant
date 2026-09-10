#!/bin/bash

# Learn in simulation (baseline MLP SAC).
if [ "$1" == "sim" ]; then
    python3 sac_cleanrl.py \
        --render_mode rgb_array \
        --env_id SimEmbodiedAnt \
        --runs_directory runs_sim_test \
        --exp_name trial_1 \
        --num_envs 1 \
        --cuda \
        --hidden_size 128
fi

if [ "$1" == "sim_hidden_dim_256" ]; then
    python3 sac_cleanrl.py \
        --render_mode rgb_array \
        --env_id SimEmbodiedAnt \
        --runs_directory runs_sim_test \
        --exp_name trial_1 \
        --num_envs 1 \
        --cuda \
        --hidden_size 256
fi

if [ "$1" == "sim_hidden_dim_512" ]; then
    python3 sac_cleanrl.py \
        --render_mode rgb_array \
        --env_id SimEmbodiedAnt \
        --runs_directory runs_sim_test \
        --exp_name trial_1 \
        --num_envs 1 \
        --cuda \
        --hidden_size 512
fi

if [ "$1" == "sim_hidden_dim_1024" ]; then
    python3 sac_cleanrl.py \
        --render_mode rgb_array \
        --env_id SimEmbodiedAnt \
        --runs_directory runs_sim_test \
        --exp_name trial_1 \
        --num_envs 1 \
        --cuda \
        --hidden_size 1024
fi

if [ "$1" == "sim_hidden_dim_2048" ]; then
    python3 sac_cleanrl.py \
        --render_mode rgb_array \
        --env_id SimEmbodiedAnt \
        --runs_directory runs_sim_test \
        --exp_name trial_1 \
        --num_envs 1 \
        --cuda \
        --hidden_size 2048
fi

# Learn in simulation with SimBa architecture.
if [ "$1" == "sim_simba_hidden_dim_128" ]; then
    python3 sac_cleanrl_simba.py \
        --render_mode rgb_array \
        --env_id SimEmbodiedAnt \
        --runs_directory runs_sim_simba_test_longer_duration \
        --exp_name trial_1_simba \
        --num_envs 1 \
        --cuda \
        --critic_hidden_dim 128 \
        --actor_hidden_dim 128
fi

if [ "$1" == "sim_simba_hidden_dim_256" ]; then
    python3 sac_cleanrl_simba.py \
        --render_mode rgb_array \
        --env_id SimEmbodiedAnt \
        --runs_directory runs_sim_simba_test_longer_duration \
        --exp_name trial_1_simba \
        --num_envs 1 \
        --cuda \
        --actor_hidden_dim 256 \
        --critic_hidden_dim 256
fi

if [ "$1" == "sim_simba_hidden_dim_512" ]; then
    python3 sac_cleanrl_simba.py \
        --render_mode rgb_array \
        --env_id SimEmbodiedAnt \
        --runs_directory runs_sim_simba_test_longer_duration \
        --exp_name trial_1_simba \
        --num_envs 1 \
        --cuda \
        --actor_hidden_dim 512 \
        --critic_hidden_dim 512
fi

if [ "$1" == "sim_simba_hidden_dim_1024" ]; then
    python3 sac_cleanrl_simba.py \
        --render_mode rgb_array \
        --env_id SimEmbodiedAnt \
        --runs_directory runs_sim_simba_test_longer_duration \
        --exp_name trial_1_simba \
        --num_envs 1 \
        --cuda \
        --actor_hidden_dim 1024 \
        --critic_hidden_dim 1024
fi

if [ "$1" == "sim_simba_hidden_dim_2048" ]; then
    python3 sac_cleanrl_simba.py \
        --render_mode rgb_array \
        --env_id SimEmbodiedAnt \
        --runs_directory runs_sim_simba_test_longer_duration \
        --exp_name trial_1_simba \
        --num_envs 1 \
        --cuda \
        --actor_hidden_dim 2048 \
        --critic_hidden_dim 2048
fi

# Learn on hardware (baseline MLP SAC).
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
        --seed 1
fi

# Learn on hardware with SimBa architecture.
if [ "$1" == "hw_simba" ]; then
    python3 sac_cleanrl_simba.py \
        --render_mode rgb_array \
        --dt 0.12 \
        --env_id HwEmbodiedAnt \
        --hw_config ../../embodied_ant_env/ant12.json \
        --learning_starts 2000 \
        --task_type back_and_forth \
        --runs_directory runs_hw_simba \
        --exp_name trial_1_simba \
        --seed 1
fi

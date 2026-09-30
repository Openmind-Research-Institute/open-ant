
# Learn on hardware.
if [ "$1" == "sim" ]; then
    python3 sarsa_options_tilecoding.py \
        --render_mode rgb_array \
        --dt 0.05 \
        --env_id SimEmbodiedAnt \
        --runs_directory runs_sarsa_sim \
        --exp_name trial_1 \
        --seed 1
fi


# Learn on hardware.
if [ "$1" == "hw" ]; then
    python3 sarsa_options_tilecoding.py \
        --render_mode rgb_array \
        --dt 0.05 \
        --env_id HwEmbodiedAnt \
        --hw_config ../../embodied_ant_env/ant12.json \
        --seed 1 \
        --runs_directory runs_sarsa_hw \
        --exp_name trial_1 \
fi
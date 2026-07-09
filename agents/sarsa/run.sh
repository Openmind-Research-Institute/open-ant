
# Learn on hardware.
if [ "$1" == "sim" ]; then
    python3 sarsa_options_tilecoding.py \
        --render_mode rgb_array \
        --dt 0.05 \
        --env_id SimEmbodiedAnt \
        --runs_directory runs_sarsa_sim_final_paper \
        --exp_name trial_1 \
        --epsilon 0.255 \
        --discount 0.998 \
        --lambda_eligibility 0.964 \
        --dim_tiling 4 \
        --tilings_multiplier 8 \
        --step_size_base 0.008 \
        --reward_scaling 5 \
        --seed 0
fi


if [ "$1" == "sim_30_seeds" ]; then
    for seed in {0..29}; do
        python3 sarsa_options_tilecoding.py \
            --render_mode rgb_array \
            --dt 0.05 \
            --env_id SimEmbodiedAnt \
            --runs_directory runs_sarsa_sim_30_seeds \
            --exp_name trial_$seed \
            --epsilon 0.26 \
            --discount 0.98 \
            --lambda_eligibility 0.94 \
            --dim_tiling 4 \
            --tilings_multiplier 8 \
            --step_size_base 0.001 \
            --reward_scaling 5 \
            --seed "$seed" &
    done

    wait
fi



if [ "$1" == "sim_30_seeds_v2" ]; then
    for seed in {0..29}; do
        python3 sarsa_options_tilecoding.py \
            --render_mode rgb_array \
            --dt 0.05 \
            --env_id SimEmbodiedAnt \
            --runs_directory runs_sarsa_sim_30_seeds_v2 \
            --exp_name trial_$seed \
            --epsilon 0.255 \
            --discount 0.998 \
            --lambda_eligibility 0.964\
            --dim_tiling 4 \
            --tilings_multiplier 8 \
            --step_size_base 0.008 \
            --reward_scaling 5 \
            --seed "$seed" &
    done

    wait
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
        --runs_directory runs_sarsa_hw_final_paper \
        --epsilon 0.255 \
        --discount 0.998 \
        --lambda_eligibility 0.964 \
        --dim_tiling 4 \
        --tilings_multiplier 8 \
        --step_size_base 0.008 \
        --reward_scaling 5
fi
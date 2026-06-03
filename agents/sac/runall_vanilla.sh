#!/bin/bash

SEEDS=(0 1 2 3 4 5 6 7 8 9 10)

for SEED in "${SEEDS[@]}"; do
    
    python3 sac_cleanrl.py \
        --render_mode rgb_array \
        --env_id SimEmbodiedAnt \
        --runs_directory runs_sim_less_aggresive/vanilla_sac \
        --exp_name trial_1 \
        --num_envs 1 \
        --radius_back_and_forth 1.0 \
        --origin_back_and_forth 0.0 0.0 \
        --seed $SEED \
        --cuda

    SIM1_DIR=$(ls -td runs_sim_less_aggresive/vanilla_sac/trial_1_2*_seed_${SEED} | grep -v continual | head -1)

    if [ -z "$SIM1_DIR" ]; then
        echo "ERROR: No sim1 folder found for seed $SEED"
        exit 1
    fi
    
    echo "Sim1 run folder: $SIM1_DIR"

    echo "Running seed $SEED: continual learning..."
 

    python3 sac_cleanrl.py \
        --render_mode rgb_array \
        --env_id SimEmbodiedAnt \
        --runs_directory runs_sim_less_aggresive/vanilla_sac \
        --exp_name trial_1_continual_learning \
        --num_envs 1 \
        --weights_path $SIM1_DIR \
        --total_timesteps 120000 \
        --model_path ../../sim/assets/ant_with_camera_after_sys_id_real_less_aggresive.xml \
        --radius_back_and_forth 1.0 \
        --origin_back_and_forth 0.0 0.0 \
        --seed $SEED \
        --cuda

    echo "Done with seed $SEED"
done

echo "All seeds complete!"
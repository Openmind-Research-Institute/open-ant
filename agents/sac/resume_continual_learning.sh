#!/bin/bash

# Map each seed to its existing sim1 folder
declare -A SIM1_DIRS
SIM1_DIRS[0]="runs_sim_less_aggresive/trial_1_20260529-021308_seed_0"
SIM1_DIRS[2]="runs_sim_less_aggresive/trial_1_20260529-022826_seed_2"
SIM1_DIRS[3]="runs_sim_less_aggresive/trial_1_20260529-024335_seed_3"
SIM1_DIRS[4]="runs_sim_less_aggresive/trial_1_20260529-025838_seed_4"
SIM1_DIRS[5]="runs_sim_less_aggresive/trial_1_20260529-031348_seed_5"
SIM1_DIRS[6]="runs_sim_less_aggresive/trial_1_20260529-032858_seed_6"
SIM1_DIRS[7]="runs_sim_less_aggresive/trial_1_20260529-034402_seed_7"
SIM1_DIRS[8]="runs_sim_less_aggresive/trial_1_20260529-035909_seed_8"
SIM1_DIRS[9]="runs_sim_less_aggresive/trial_1_20260529-041414_seed_9"
SIM1_DIRS[10]="runs_sim_less_aggresive/trial_1_20260529-042924_seed_10"

SEEDS=(0 2 3 4 5 6 7 8 9 10)

for SEED in "${SEEDS[@]}"; do
    SIM1_DIR=${SIM1_DIRS[$SEED]}
    echo "========================================="
    echo "Running seed $SEED: continual learning..."
    echo "Sim1 dir: $SIM1_DIR"
    echo "========================================="

    python3 sac_cleanrl.py \
        --render_mode rgb_array \
        --env_id SimEmbodiedAnt \
        --runs_directory runs_sim_less_aggresive \
        --exp_name trial_1_continual_learning \
        --num_envs 1 \
        --weights_path $SIM1_DIR \
        --offline_buffer_path $SIM1_DIR/replay_buffer \
        --total_timesteps 120000 \
        --model_path ../../sim/assets/ant_with_camera_after_sys_id_real_less_aggresive.xml \
        --radius_back_and_forth 1.0 \
        --origin_back_and_forth 0.0 0.0 \
        --seed $SEED \
        --cuda

    echo "Done with seed $SEED"
done

echo "All seeds complete!"

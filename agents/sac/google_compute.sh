#!/bin/bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

for seed in {1..8}; do
    core0=$(( (seed-1)*2 ))
    core1=$(( core0+1 ))

    taskset -c ${core0},${core1} \
    python3 sac_cleanrl.py \
        --seed ${seed} \
        --exp_name sac_seed_${seed} \
        --total_timesteps 1000000 \
        --env_id SimEmbodiedAnt \
        --task_type back_and_forth \
        --model_path ../../sim/assets/ant_with_camera_after_sys_id.xml \
        --runs_directory runs_parallel &
done

wait
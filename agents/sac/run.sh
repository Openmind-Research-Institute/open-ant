#!/bin/bash

# learn in sim
if [ "$1" == "sim" ]; then
    python3 sac_cleanrl.py --render_mode rgb_array --dt 0.12 --env_id SimEmbodiedAnt --learning_starts 2000
fi

if [ "$1" == "hw" ]; then
    python3 sac_cleanrl.py --render_mode rgb_array --dt 0.12 --env_id HwEmbodiedAnt --hw_config ../../embodied_ant_env/ant12.json --learning_starts 2000
fi


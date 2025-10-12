#!/bin/bash

# learn in sim
if [ "$1" == "sim" ]; then
    python3 sac_cleanrl.py --render_mode rgb_array --capture_video --dt 0.1 --env_id SimEmbodiedAnt --learning_starts 2000
fi

# # learn in hardware
# if [ "$1" == "hw" ]; then
#     python3 sac_cleanrl.py --render_mode rgb_array --capture_video --dt 0.1 --env_id HwEmbodiedAnt --hw_config /Users/sorinalupu/OpenmindResearch/workshops/EmbodiedAnt/embodied_ant_env/ant34.json --weights_path /Users/sorinalupu/OpenmindResearch/workshops/EmbodiedAnt/agents/sac_skrl/runs/SimEmbodiedAnt__sac_cleanrl__1__20251011-124337/weights --learning_starts 20
# fi

if [ "$1" == "hw" ]; then
    python3 sac_cleanrl.py --render_mode rgb_array --dt 0.1 --env_id HwEmbodiedAnt --hw_config /Users/sorinalupu/OpenmindResearch/workshops/EmbodiedAnt/embodied_ant_env/ant34.json --learning_starts 2000
fi


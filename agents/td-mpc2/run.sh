#!/bin/bash

# learn in sim
if [ "$1" == "sim" ]; then
    python3 learn.py exp_name=sim
fi

if [ "$1" == "hw" ]; then
    python3 learn.py hw_config="/home/sorina/embodied-mujoco-ant/embodied_ant_env/ant12.json" exp_name=hw
fi


#!/bin/bash

python sac_skrl.py --train True --seed 0 --terminate_when_upside_down True --upside_down_cost_weight 0.0 --ctrl_cost_weight 0.0
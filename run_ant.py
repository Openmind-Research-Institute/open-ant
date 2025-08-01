#!/usr/bin/env python3
"""
Example script to run the embodied ant environment.
"""

import sys
import os
import json
import numpy as np
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'embodied_ant_env')))
from embodied_ant_env import make_ant_env

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_ant.py <config_file.json>")
        sys.exit(1)

    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        cfg = json.load(f)
    env = make_ant_env(cfg, render_mode='human')
    
    try:
        i = 0
        while True:
            obs, rew, term, trunc, info = env.step(np.zeros(8))
            print(f"Reward: {rew}")

    except KeyboardInterrupt:
        print("\nShutting down...")
        time.sleep(0.1)
    finally:
        env.close()

if __name__ == "__main__":
    main()

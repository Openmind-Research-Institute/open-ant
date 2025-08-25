import os
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=100)

from datetime import datetime
import functools
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from IPython.display import clear_output
from matplotlib import pyplot as plt
import numpy as np
from ml_collections import config_dict
from wrapper import wrap_for_brax_training

import time
from etils import epath
import jax
import mujoco

from tqdm import tqdm
import mediapy as media
import argparse

# Local imports.
from ant import Ant

# Set seed.
SEED = 0
rng = jax.random.PRNGKey(SEED)


parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=False)
args = parser.parse_args()
print('Training: ', args.train)

if args.train == True:
  # Folders.
  RESULTS = 'results'
  if not os.path.exists(RESULTS):
      os.makedirs(RESULTS)
  time_now = datetime.now().strftime('%Y%m%d-%H%M%S')
  if not os.path.exists(os.path.join(RESULTS, time_now)):
      os.makedirs(os.path.join(RESULTS, time_now))
  FOLDER_RESULTS = os.path.join(RESULTS, time_now)
  ABS_FOLDER_RESUlTS = os.path.abspath(FOLDER_RESULTS)
  print(f"Saving results to {ABS_FOLDER_RESUlTS}")
  print("Available devices:", jax.devices())

  # Brax PPO config.
  brax_ppo_config = config_dict.create(
        num_timesteps=200_000_000,
        num_evals=15,
        reward_scaling=1.0,
        clipping_epsilon=0.2,
        num_resets_per_eval=1,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=0.005,
        num_envs=8192,
        batch_size=256,
        max_grad_norm=1.0,
        network_factory = config_dict.create(
          policy_hidden_layer_sizes=(512, 256, 128),
          value_hidden_layer_sizes=(512, 256, 128),
          policy_obs_key="state",
          value_obs_key="state",
        ),
    )
  ppo_params = brax_ppo_config

  # Environment.
  env = Ant(save_config_folder=ABS_FOLDER_RESUlTS)
  eval_env = Ant(save_config_folder=ABS_FOLDER_RESUlTS)

  x_data, y_data, y_dataerr = [], [], []
  times = [datetime.now()]
  reward_list = []
  def progress(num_steps, metrics):
    clear_output(wait=True)

    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    y_dataerr.append(metrics["eval/episode_reward_std"])

    reward_list.append([num_steps, metrics["eval/episode_reward"]])

    _, ax = plt.subplots()
    ax.set_xlim([0, ppo_params["num_timesteps"] * 1.25])
    ax.set_xlabel("# environment steps")
    ax.set_ylabel("reward per episode")
    ax.set_title(f"y={y_data[-1]:.3f}")
    ax.plot(x_data, y_data)
    ax.fill_between(x_data, np.array(y_data) - np.array(y_dataerr), np.array(y_data) + np.array(y_dataerr), alpha=0.2)
    plt.savefig(f'{ABS_FOLDER_RESUlTS}/reward.pdf')
    plt.savefig(f'{ABS_FOLDER_RESUlTS}/reward.png')
    print("Reward for {} steps: {:.3f}".format(num_steps, y_data[-1]))
    
  ppo_training_params = dict(ppo_params)
  network_factory = ppo_networks.make_ppo_networks
  if "network_factory" in ppo_params:
    del ppo_training_params["network_factory"]
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **ppo_params.network_factory
    )

  # Train the policy.
  train_fn = functools.partial(
      ppo.train, **dict(ppo_training_params),
      network_factory=network_factory,
      progress_fn=progress,
      save_checkpoint_path=ABS_FOLDER_RESUlTS,
  )

  make_inference_fn, params, metrics = train_fn(
      environment=env,
      eval_env=eval_env,
      wrap_env_fn=wrap_for_brax_training,
  )
  print(f"time to jit: {times[1] - times[0]}")
  print(f"time to train: {times[-1] - times[1]}")

else:
  # Testing: load the latest weights and test the policy.
  RESULTS_FOLDER_PATH = os.path.abspath('results')
  folders = sorted(os.listdir(RESULTS_FOLDER_PATH))
  latest_folder = folders[-1]
  folders = sorted(os.listdir(epath.Path(RESULTS_FOLDER_PATH) / latest_folder))
  folders = [f for f in folders if os.path.isdir(epath.Path(RESULTS_FOLDER_PATH) / latest_folder / f)]

  latest_weights_folder = folders[-1]
  print(f'Latest weights folder: {latest_weights_folder}')
  policy_fn = ppo_checkpoint.load_policy(epath.Path(RESULTS_FOLDER_PATH) / latest_folder / latest_weights_folder)

  eval_env = Ant()
  jit_reset = jax.jit(eval_env.reset)
  jit_step = jax.jit(eval_env.step)
  jit_policy = jax.jit(policy_fn)
  step_fn = jax.jit(eval_env.step)
  rollout = []
  modify_scene_fns = []

  state = jit_reset(rng)

  metrics_list = []
  ctrl_list = []
  state_list = []
  for i in tqdm(range(1400)):
    act_rng, rng = jax.random.split(rng)
    delta_ctrl, _ = jit_policy(state.obs, act_rng)
    ctrl_list.append(delta_ctrl)
    state = jit_step(state, delta_ctrl)
    state_list.append(state.obs["state"])
    metrics_list.append(state.metrics)
    if state.done:
      break
    rollout.append(state)

  render_every = 1
  fps = 1.0 / eval_env.ctrl_dt / render_every
  traj = rollout[::render_every]
  # mod_fns = modify_scene_fns[::render_every]

  scene_option = mujoco.MjvOption()
  scene_option.geomgroup[2] = True
  scene_option.geomgroup[3] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False

  frames = eval_env.render(
      traj,
      camera="track",
      scene_option=scene_option,
      width=640,
      height=480,
      # modify_scene_fns=mod_fns,
  )
  
  media.write_video(f'{epath.Path(RESULTS_FOLDER_PATH) / latest_folder / latest_weights_folder}/ant.mp4', frames, fps=fps)
  print('Video saved')
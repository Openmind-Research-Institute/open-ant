from typing import Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx
from flax import struct

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

def get_rz(
    phi: Union[jax.Array, float], 
    swing_height: Union[jax.Array, float] = 0.08
) -> jax.Array:
  def cubic_bezier_interpolation(y_start, y_end, x):
    y_diff = y_end - y_start
    bezier = x**3 + 3 * (x**2 * (1 - x))
    return y_start + y_diff * bezier

  x = (phi + jp.pi) / (2 * jp.pi)
  stance = cubic_bezier_interpolation(0, swing_height, 2 * x)
  swing = cubic_bezier_interpolation(swing_height, 0, 2 * x - 1)
  return jp.where(x <= 0.5, stance, swing)


def get_collision_info(
    contact: Any, geom1: int, geom2: int
) -> Tuple[jax.Array, jax.Array]:
  """Get the distance and normal of the collision between two geoms."""
  mask = (jp.array([geom1, geom2]) == contact.geom).all(axis=1)
  mask |= (jp.array([geom2, geom1]) == contact.geom).all(axis=1)
  idx = jp.where(mask, contact.dist, 1e4).argmin()
  dist = contact.dist[idx] * mask[idx]
  normal = (dist < 0) * contact.frame[idx, 0, :3]
  return dist, normal


def geoms_colliding(state: mjx.Data, geom1: int, geom2: int) -> jax.Array:
  """Return True if the two geoms are colliding."""
  return get_collision_info(state.contact, geom1, geom2)[0] < 0

def draw_joystick_command(
    scn,
    cmd,
    xyz,
    theta,
    rgba=None,
    radius=0.02,
    scl=1.0,
):
  if rgba is None:
    rgba = [0.2, 0.2, 0.6, 0.3]
  scn.ngeom += 1
  scn.geoms[scn.ngeom - 1].category = mujoco.mjtCatBit.mjCAT_DECOR

  vx, vy, vtheta = cmd

  angle = theta + vtheta
  rotation_matrix = np.array(
      [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
  )

  arrow_from = xyz
  rotated_velocity = rotation_matrix @ np.array([vx, vy])
  to = np.asarray([rotated_velocity[0], rotated_velocity[1], 0])
  to = to / (np.linalg.norm(to) + 1e-6)
  arrow_to = arrow_from + to * scl

  mujoco.mjv_initGeom(
      geom=scn.geoms[scn.ngeom - 1],
      type=mujoco.mjtGeom.mjGEOM_ARROW.value,
      size=np.zeros(3),
      pos=np.zeros(3),
      mat=np.zeros(9),
      rgba=np.asarray(rgba).astype(np.float32),
  )
  mujoco.mjv_connector(
      geom=scn.geoms[scn.ngeom - 1],
      type=mujoco.mjtGeom.mjGEOM_ARROW.value,
      width=radius,
      from_=arrow_from,
      to=arrow_to,
  )

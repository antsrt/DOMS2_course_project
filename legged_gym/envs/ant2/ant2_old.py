# Copyright 2023 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint:disable=g-multiple-import
"""Trains an ant to run in the +x direction."""

from typing import Any
import torch
from legged_gym.envs.base.legged_robot import LeggedRobot
from .ant2_config import Ant2RoughCfg


class Ant2(LeggedRobot):

    def __init__(self, cfg: Ant2RoughCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self._ctrl_cost_weight = 0.5
        self._healthy_reward = 0.0
        self._terminate_when_unhealthy = True
        self._healthy_z_range = (0.2, 1.0)
        self._reset_noise_scale = 0.1
        self._exclude_current_positions_from_observation = True

    def _get_obs(self):
        """Observe ant body position and velocities."""
        # Match Brax observation: z, quat, joint_pos, lin_vel, ang_vel, joint_vel
        z = self.root_states[:, 2:3]
        quat = self.root_states[:, 3:7]
        joint_pos = self.dof_pos
        vel = self.root_states[:, 7:13]  # lin_vel + ang_vel
        joint_vel = self.dof_vel
        return torch.cat([z, quat, joint_pos, vel, joint_vel], dim=1)

    def compute_observations(self):
        self.obs_buf = self._get_obs()
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def is_done(self):
        """Returns the done signal."""
        done = (1.0 - self._is_healthy() if self._terminate_when_unhealthy else 0.0)
        return done

    def _is_healthy(self):
        """Returns the healthy signal."""
        if self._exclude_current_positions_from_observation:
            z = self.root_states[:, 2]
        else:
            z = self.root_states[:, 2]
        min_z, max_z = self._healthy_z_range
        is_healthy = torch.where(z < min_z, 0.0, 1.0)
        is_healthy = torch.where(z > max_z, 0.0, is_healthy)
        return is_healthy

    def compute_reward(self):
        if self._exclude_current_positions_from_observation:
            velocity = self.obs_buf[:, 13:16]
        else:
            velocity = self.obs_buf[:, 15:18]
        forward_reward = 1.0 * velocity[:, 0]  # increased reward for forward movement

        is_healthy = self._is_healthy()
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy

        ctrl_cost = self._ctrl_cost_weight * torch.sum(torch.square(self.actions), dim=1)
        contact_cost = 0.0

        # Add penalties from config scales
        torques_penalty = self.cfg.rewards.scales.torques * torch.sum(torch.square(self.torques), dim=1)
        # DOF position limits penalty
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        dof_pos_penalty = self.cfg.rewards.scales.dof_pos_limits * torch.sum(out_of_limits, dim=1)

        reward = forward_reward + healthy_reward - ctrl_cost - contact_cost - torques_penalty - dof_pos_penalty

        # Update episode sums for logging
        self.episode_sums['torques'] += torques_penalty
        self.episode_sums['dof_pos_limits'] += dof_pos_penalty

        # write reward to buffer used by the runner
        # ensure shape matches (num_envs,)
        try:
            self.rew_buf = reward
        except Exception:
            self.rew_buf[:] = reward

        # apply optional clipping to keep compatibility with other envs
        if hasattr(self.cfg.rewards, 'only_positive_rewards') and self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)

        return reward

    def _compute_reward(self):
        return self.compute_reward()

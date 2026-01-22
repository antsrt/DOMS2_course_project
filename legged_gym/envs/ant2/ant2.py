# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import torch
from isaacgym.torch_utils import torch_rand_float
from legged_gym.envs.base.legged_robot import LeggedRobot

class Ant2(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.cot = torch.zeros(self.num_envs, device=self.device)
        self.cot_sum = torch.zeros(self.num_envs, device=self.device)

    def _compute_torques(self, actions):
        torques = super()._compute_torques(actions)
        
                # Сохраняем для последующего использования в награде
        self.instant_power = torch.abs(torques * self.dof_vel)
        
        # Суммарная мощность по всем суставам для каждого окружения
        self.power_total = torch.sum(self.instant_power, dim=1)
        
        # Debug: print torques and dof_pos for first env
        if not self.headless and self.common_step_counter % 100 == 0:
            print(f"Actions: {actions[0]}")
            print(f"Torques: {torques[0]}")
            print(f"DOF pos: {self.dof_pos[0]}")
            print(f"DOF vel: {self.dof_vel[0]}")
        return torques

    def compute_observations(self):
        """ Computes observations for Ant: 38 dims with commands (vx, vy, yaw)
        """
        self.obs_buf = torch.cat((
            self.root_states[:, 2:3],  # z position
            self.base_quat,  # w, x, y, z quaternion
            self.dof_pos,  # 8 joint positions
            self.base_lin_vel,  # x, y, z velocities
            self.base_ang_vel,  # angular velocities
            self.commands[:, :3],  # vx, vy, yaw command
            self.dof_vel  # 8 joint velocities
        ), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def check_termination(self):
        """ Check terminations for Ant
        """
        # termination on contact forces on torso
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        # termination on unhealthy (z out of range)
        self.reset_buf |= ~self._is_healthy()
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    def _is_healthy(self):
        """Returns the healthy signal based on z position."""
        min_z, max_z = 0.2, 1.0
        z = self.root_states[:, 2]
        is_healthy = (z >= min_z) & (z <= max_z)
        return is_healthy

    def _reward_tracking_lin_vel(self):
        # reward for tracking linear velocity command in world frame (x, y)
        lin_vel_world = self.root_states[:, 7:9]
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - lin_vel_world), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_torques(self):
        # penalty term for large torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_collision(self):
        # penalty term for contact forces
        contact_force = torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1)
        clipped_force = torch.clamp(contact_force, -1., 1.)
        return 0.5 * 0.001 * torch.sum(torch.square(clipped_force), dim=1)

    def _reward_survive(self):
        # reward for surviving each timestep
        return 1.0

    def _reward_dof_pos_limits(self):
        # penalty for exceeding soft dof position limits
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_energy(self):
        # reward = self.gait_cfg.reward.alpha_en * torch.exp(-(torch.sum(abs(self.robot_env.dof_vel) * abs(self.robot_env.torques), dim=1)) / (self.gait_cfg.reward.sigma_en_x * abs(self.robot_env.base_lin_vel[:, 0]) + self.gait_cfg.reward.sigma_en_z * abs(self.robot_env.base_ang_vel[:, 2])))
        lin_vel = torch.abs(self.root_states[:, 7])
        ang_vel = torch.abs(self.root_states[:, 12])
        denom = 400 * lin_vel + 70 * ang_vel
        reward = torch.exp(-self.power_total / (denom + 1e-6))
        mg = 200
        self.cot = self.power_total / (mg * lin_vel + 1e-6)
        self.cot_sum += self.cot
        return reward
    
    def _resample_commands(self, env_ids):
        """ Resample commands for Ant: lin_vel_x, lin_vel_y, ang_vel_yaw, heading
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)

    def _reward_tracking_ang_vel(self):
        # reward for tracking angular velocity (yaw) in world frame
        ang_vel_world = self.root_states[:, 12]
        ang_vel_error = torch.square(self.commands[:, 2] - ang_vel_world)
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_smooth_gait(self):
        # Smoothness reward based on normalized action deltas.
        delta = self.actions - self.last_actions
        scaled_delta = delta / 2.0  # policy outputs are typically in [-1, 1]
        return torch.exp(-(scaled_delta ** 2).sum(dim=1))

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        cot_mean = self.cot_sum[env_ids] / torch.clamp(self.episode_length_buf[env_ids].float(), min=1.0)
        super().reset_idx(env_ids)
        self.extras["episode"]["cot"] = torch.mean(cot_mean)
        self.cot_sum[env_ids] = 0.0

    def _get_noise_scale_vec(self, cfg):
        """ Sets noise for Ant observations
        """
        noise_vec = torch.zeros(38, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        # z pos
        noise_vec[0] = noise_scales.height_measurements * noise_level
        # quat
        noise_vec[1:5] = 0.  # no noise on orientation?
        # joint pos
        noise_vec[5:17] = noise_scales.dof_pos * noise_level
        # lin vel
        noise_vec[17:20] = noise_scales.lin_vel * noise_level
        # ang vel
        noise_vec[20:23] = noise_scales.ang_vel * noise_level
        # commands
        noise_vec[23:26] = 0.
        # joint vel
        noise_vec[26:38] = noise_scales.dof_vel * noise_level
        return noise_vec

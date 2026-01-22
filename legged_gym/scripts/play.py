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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import matplotlib.pyplot as plt
from multiprocessing import Process
import csv


def build_trapezoid_profile(total_steps, v_start, v_peak, ramp_fraction=0.25):
    if total_steps <= 0:
        return np.array([], dtype=np.float32)
    if total_steps == 1:
        return np.array([v_peak], dtype=np.float32)
    ramp_fraction = max(0.01, min(0.49, float(ramp_fraction)))
    ramp_steps = max(1, int(round(total_steps * ramp_fraction)))
    if 2 * ramp_steps > total_steps:
        ramp_steps = max(1, total_steps // 2)
    plateau_steps = max(0, total_steps - 2 * ramp_steps)
    profile = np.empty(total_steps, dtype=np.float32)
    for step in range(total_steps):
        if step < ramp_steps:
            denom = max(1, ramp_steps - 1)
            alpha = step / denom
            value = v_start + (v_peak - v_start) * alpha
        elif step < ramp_steps + plateau_steps:
            value = v_peak
        else:
            down_step = step - ramp_steps - plateau_steps
            denom = max(1, ramp_steps - 1)
            alpha = down_step / denom
            value = v_peak - (v_peak - v_start) * alpha
        profile[step] = value
    return profile


def plot_cot_vs_speed(speeds, cot_values, skip, bin_step=0.05, csv_path=None):
    if not cot_values or not speeds:
        return
    skip = max(0, int(skip))
    min_len = min(len(cot_values), len(speeds))
    if skip >= min_len:
        skip = max(0, min_len - 1)
    cot_filtered = cot_values[skip:min_len]
    speed_filtered = speeds[skip:min_len]
    if not cot_filtered or not speed_filtered:
        return
    speeds_arr = np.asarray(speed_filtered, dtype=np.float32)
    cot_arr = np.asarray(cot_filtered, dtype=np.float32)
    fig, ax = plt.subplots()
    bin_step = max(1e-6, float(bin_step))
    min_speed = float(np.min(speeds_arr))
    max_speed = float(np.max(speeds_arr))
    csv_rows = []
    if max_speed <= min_speed:
        single_mean = float(np.mean(cot_arr))
        ax.plot([min_speed], [single_mean], marker='o', label='cot')
        csv_rows.append((min_speed, single_mean, int(cot_arr.size)))
    else:
        bin_edges = np.arange(min_speed, max_speed + bin_step, bin_step, dtype=np.float32)
        if bin_edges.size < 2:
            bin_edges = np.array([min_speed, max_speed + bin_step], dtype=np.float32)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        bin_ids = np.digitize(speeds_arr, bin_edges) - 1
        centers = []
        means = []
        for i in range(len(bin_centers)):
            mask = bin_ids == i
            if not np.any(mask):
                continue
            center = float(bin_centers[i])
            mean_val = float(np.mean(cot_arr[mask]))
            count = int(np.sum(mask))
            centers.append(center)
            means.append(mean_val)
            csv_rows.append((center, mean_val, count))
        if not centers:
            return
        ax.plot(centers, means, marker='o', label=f'cot (mean per {bin_step:.2f} m/s)')
    if csv_rows:
        mean_cot = float(np.mean([row[1] for row in csv_rows]))
    else:
        mean_cot = float(np.mean(cot_arr))
    ax.axhline(mean_cot, color='r', linestyle='--', label=f'mean {mean_cot:.3f}')
    ax.set(xlabel='speed [m/s]', ylabel='CoT', title='Cost of Transport vs Speed')
    ax.legend()
    fig.tight_layout()
    if csv_path and csv_rows:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["speed", "cot_mean", "count"])
            writer.writerows(csv_rows)
    plt.show()


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # Set custom target speeds (modify these values as needed)
    desired_lin_vel_y = 0.0  # m/s sideways
    desired_ang_vel_yaw = 0.0  # rad/s rotation
    lin_vel_x_start = 0.2
    
    # Disable command resampling to keep fixed speeds
    env.cfg.commands.resampling_time = 1e9  # large number to prevent resampling
    
    lin_vel_x_min, lin_vel_x_max = env_cfg.commands.ranges.lin_vel_x
    lin_vel_x_min, lin_vel_x_max = sorted([float(lin_vel_x_min), float(lin_vel_x_max)])
    lin_vel_x_start = max(lin_vel_x_min, min(lin_vel_x_start, lin_vel_x_max))
    lin_vel_x_peak = max(lin_vel_x_start, lin_vel_x_max)
    total_steps = max(1, 10 * int(env.max_episode_length))
    ramp_fraction = 0.15
    lin_vel_x_profile = build_trapezoid_profile(
        total_steps,
        lin_vel_x_start,
        lin_vel_x_peak,
        ramp_fraction=ramp_fraction,
    )
    if lin_vel_x_profile.size == 0:
        lin_vel_x_profile = np.full(total_steps, lin_vel_x_start, dtype=np.float32)

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 1000 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    exp_name = args.experiment_name if args.experiment_name else train_cfg.runner.experiment_name
    cot_log = []
    speed_log = []
    cot_skip = 10
    for i in range(total_steps):
        desired_lin_vel_x = float(lin_vel_x_profile[i])
        env.commands[:, 0] = desired_lin_vel_x
        env.commands[:, 1] = desired_lin_vel_y
        env.commands[:, 2] = desired_ang_vel_yaw
        env.compute_observations()
        obs = env.get_observations()
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            if hasattr(env, "power_total"):
                lin_vel = abs(env.root_states[robot_index, 7].item())
                cot_val = (env.power_total[robot_index].item() / (200.0 * lin_vel + 1e-6)) * 100
            else:
                lin_vel = 0.0
                cot_val = 0.0
            cot_log.append(cot_val)
            speed_log.append(lin_vel)
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale
                    + env.default_dof_pos[0, joint_index].item(),
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
            if cot_log:
                csv_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", exp_name, f"{exp_name}.csv")
                Process(target=plot_cot_vs_speed, args=(speed_log, cot_log, cot_skip, 0.05, csv_path)).start()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)

# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions and the following disclaimer in the documentation
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

from isaacgym import gymapi
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Ant2RoughCfg( LeggedRobotCfg ):
    class env(LeggedRobotCfg.env):
        num_envs = 100
        num_observations = 27
        num_privileged_obs = None
        num_actions = 8
        env_spacing = 3.
        send_timeouts = True
        episode_length_s = 20

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        horizontal_scale = 0.1
        vertical_scale = 0.005
        curriculum = False
        measure_heights = False

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        num_commands = 4
        resampling_time = 10.
        heading_command = True
        class ranges:
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1, 1]
            heading = [-3.14, 3.14]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 1.0]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        default_joint_angles = {
            'hip_1': 0.0,
            'ankle_1': 0.0,
            'hip_2': 0.0,
            'ankle_2': 0.0,
            'hip_3': 0.0,
            'ankle_3': 0.0,
            'hip_4': 0.0,
            'ankle_4': 0.0,
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        # For position control, stiffness and damping are used
        stiffness = {
            'hip_1': 100.0,
            'ankle_1': 100.0,
            'hip_2': 100.0,
            'ankle_2': 100.0,
            'hip_3': 100.0,
            'ankle_3': 100.0,
            'hip_4': 100.0,
            'ankle_4': 100.0,
        }
        damping = {
            'hip_1': 1.0,
            'ankle_1': 1.0,
            'hip_2': 1.0,
            'ankle_2': 1.0,
            'hip_3': 1.0,
            'ankle_3': 1.0,
            'hip_4': 1.0,
            'ankle_4': 1.0,
        }
        # adjust action scale for position control
        action_scale = 0.5
        # adjust decimation for control update frequency
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/isaacgym/assets/mjcf/nv_ant.xml'
        name = "ant"

        foot_name = "foot"
        penalize_contacts_on = ["aux_1", "aux_2", "aux_3", "aux_4"]
        terminate_after_contacts_on = ["torso"]

        default_dof_drive_mode = 1  # position mode for position control
        collapse_fixed_joints = True
        replace_cylinder_with_capsule = True
        flip_visual_attachments = False

        density = 1000.0
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 1000.0
        max_linear_velocity = 1000.0
        armature = 0.0
        self_collisions = 1

    class rewards:
        # allow negative total rewards (don't clip to zero)
        only_positive_rewards = False
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales:
            # increase torque penalty to discourage overly aggressive torques
            torques = -0.0005
            dof_pos_limits = -1.0

class Ant2RoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.2  # reduced initial noise
    class algorithm:
        # PPO hyperparameters tuned for stability
        learning_rate = 5e-4  # increased learning rate
        clip_param = 0.1
        num_learning_epochs = 8
        num_mini_batches = 8
        entropy_coef = 0.005
        max_grad_norm = 0.5
        use_clipped_value_loss = True
        value_loss_coef = 1.0
        gamma = 0.99
        lam = 0.95
    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        # increase steps per env to reduce variance per update
        num_steps_per_env = 64
        max_iterations = 10000
        save_interval = 50
        resume = False
        load_run = -1
        checkpoint = -1
        run_name = ''
        experiment_name = 'rough_ant2'
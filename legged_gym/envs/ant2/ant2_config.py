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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Ant2Cfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 100
        num_observations = 38
        num_actions = 12
        episode_length_s = 20  # increased episode length for longer training

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane'  # flat terrain for ant
        static_friction = 2.0  # increased friction for better grip
        dynamic_friction = 2.0
        restitution = 0.0
        curriculum = False

    class commands( LeggedRobotCfg.commands ):
        num_commands = 4  # lin_vel_x, lin_vel_y, ang_vel_yaw, heading
        resampling_time = 10.
        heading_command = False
        class ranges:
            lin_vel_x = [0.5, 1.2]  # desired forward speed range
            lin_vel_y = [-0.2, 0.2]  # decreased speed for training
            ang_vel_yaw = [-0.2, 0.2]  # decreased angular speed for training
            heading = [-3.14, 3.14]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.5]  # start higher
        default_joint_angles = {
            'hip_1': 0.0,
            'hip_1_pitch': 0.0,
            'ankle_1': 0.0,
            'hip_2': 0.0,
            'hip_2_pitch': 0.0,
            'ankle_2': 0.0,
            'hip_3': 0.0,
            'hip_3_pitch': 0.0,
            'ankle_3': 0.0,
            'hip_4': 0.0,
            'hip_4_pitch': 0.0,
            'ankle_4': 0.0,
        }

    class control( LeggedRobotCfg.control ):
        control_type = 'P'  # position control
        # PD Drive parameters:
        stiffness = {'hip_1': 20.0, 'ankle_1': 10.0, 'hip_2': 20.0, 'ankle_2': 10.0, 'hip_3': 20.0, 'ankle_3': 10.0, 'hip_4': 20.0, 'ankle_4': 10.0}  # reduced stiffness for ankles to encourage their use
        damping = {'hip_1': 0.5, 'ankle_1': 0.5, 'hip_2': 0.5, 'ankle_2': 0.5, 'hip_3': 0.5, 'ankle_3': 0.5, 'hip_4': 0.5, 'ankle_4': 0.5}  # increased damping
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.2  # increased to allow more joint movement, encouraging ankle use
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        # increased joint angle limits
        dof_pos_limits = [[-1.0, 1.0] for _ in range(8)]

    class asset( LeggedRobotCfg.asset ):
        #file = "{LEGGED_GYM_ROOT_DIR}/isaacgym/assets/mjcf/nv_ant_thigh_long.xml"

        # file = "{LEGGED_GYM_ROOT_DIR}/isaacgym/assets/mjcf/nv_ant_short.xml"
        file = "{LEGGED_GYM_ROOT_DIR}/isaacgym/assets/mjcf/nv_ant_long.xml"
        #file = "{LEGGED_GYM_ROOT_DIR}/isaacgym/assets/mjcf/nv_ant_front_short_back_long.xml"
        #file = "{LEGGED_GYM_ROOT_DIR}/isaacgym/assets/mjcf/nv_ant_front_long_back_short.xml"
        #file = "{LEGGED_GYM_ROOT_DIR}/isaacgym/assets/mjcf/nv_ant.xml"  # assuming mjcf can be used or convert to urdf
        name = "ant2"
        foot_name = "foot"
        penalize_contacts_on = ["aux_1", "aux_2", "aux_3", "aux_4"]
        terminate_after_contacts_on = ["torso"]
        self_collisions = 1
        default_dof_drive_mode = 1  # position mode
        density = 2000.0  # increased density to make the robot heavier

    class rewards( LeggedRobotCfg.rewards ):
        base_height_target = 0.6
        soft_dof_pos_limit = 0.9
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 5.0  # reward_forward
            tracking_ang_vel = 0.5  # enable tracking of angular velocity
            torques = -0.0001  # reduced torque penalty
            collision = -0.5 * 0.001  # contact_cost
            survive = 1.0  # reward_survive
            dof_pos_limits = -1.0  # penalty for dof limits
            base_height = -1.0  # increased penalty for base height deviation
            energy = 1.0
            action_rate = 0.0  # disabled; using smooth_gait reward
            smooth_gait = 0.1  # exp reward for smooth actions

class Ant2CfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'ant2'
        max_iterations = 1000  # number of training iterations
        save_interval = 50  # save model every 50 iterations

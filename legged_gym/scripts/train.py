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
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    cot_log = None
    if ppo_runner.log_dir is not None:
        cot_log_path = os.path.join(ppo_runner.log_dir, "cot.log")
        os.makedirs(ppo_runner.log_dir, exist_ok=True)
        cot_log = open(cot_log_path, "a", encoding="utf-8")
        cot_log.write("# it, mean_episode_cot\n")

        original_log = ppo_runner.log

        def log_with_cot(locs, width=80, pad=35):
            original_log(locs, width=width, pad=pad)
            if not locs.get("ep_infos"):
                return
            cot_values = []
            for ep_info in locs["ep_infos"]:
                if "cot" in ep_info:
                    cot_val = ep_info["cot"]
                    if isinstance(cot_val, torch.Tensor):
                        cot_val = cot_val.item() if cot_val.numel() == 1 else cot_val.mean().item()
                    cot_values.append(float(cot_val))
            if cot_values:
                mean_cot = float(np.mean(cot_values))
                cot_log.write(f"{locs.get('it')},{mean_cot}\n")

        ppo_runner.log = log_with_cot

    try:
        ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    finally:
        if cot_log is not None:
            cot_log.close()

if __name__ == '__main__':
    args = get_args()
    train(args)

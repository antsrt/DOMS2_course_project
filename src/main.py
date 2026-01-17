import os
import numpy as np
from itertools import product

import mujoco
from mujoco import viewer 

from generate_model import generate_model

base_model_path = "model/base_model.xml"

front_leg_len_range = [0.3, 0.9, 0.1]
front_ankle_len_range = [0.9, 1.5, 0.1]
back_leg_len_range = [0.4, 0.7, 0.1]
back_ankle_len_range = [0.1, 0.3, 0.1]

def frange(start, stop, step):
    return np.arange(start, stop + step/2, step).round(10)

params = product(
    frange(*front_leg_len_range),
    frange(*front_ankle_len_range),
    frange(*back_leg_len_range),
    frange(*back_ankle_len_range)
)

def run_scene(model_path):
    if not os.path.isfile(model_path):
        raise SystemExit(f"Model file not found: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    with viewer.launch(model, data) as v:
        while v.is_running():
            mujoco.mj_step(model, data)
            v.sync()

if __name__ == "__main__":
    # for front_leg_len, front_ankle_len, back_leg_len, back_ankle_len in params:
    #     res_model_path = f"res_model/model_{front_leg_len}_{front_ankle_len}_{back_leg_len}_{back_ankle_len}.xml"

    #     generate_model(base_model_path, res_model_path, front_leg_len, front_ankle_len, back_leg_len, back_ankle_len)

    # run_scene("res_model/model_0.9_1.0_0.7_0.3.xml")

    # # 1
    # front_leg_len = 1
    # front_ankle_len = 1
    # back_leg_len = 1
    # back_ankle_len = 1

    # res_model_path = f"res_model/model_{front_leg_len}_{front_ankle_len}_{back_leg_len}_{back_ankle_len}.xml"

    # generate_model(base_model_path, res_model_path, front_leg_len, front_ankle_len, back_leg_len, back_ankle_len)

    # # 2
    # front_leg_len = 5
    # front_ankle_len = 2
    # back_leg_len = 1
    # back_ankle_len = 1

    # res_model_path = f"res_model/model_{front_leg_len}_{front_ankle_len}_{back_leg_len}_{back_ankle_len}.xml"

    # generate_model(base_model_path, res_model_path, front_leg_len, front_ankle_len, back_leg_len, back_ankle_len)

    # # 3
    # front_leg_len = 2
    # front_ankle_len = 5
    # back_leg_len = 1
    # back_ankle_len = 1

    # res_model_path = f"res_model/model_{front_leg_len}_{front_ankle_len}_{back_leg_len}_{back_ankle_len}.xml"

    # generate_model(base_model_path, res_model_path, front_leg_len, front_ankle_len, back_leg_len, back_ankle_len)

    # # 4
    # front_leg_len = 1
    # front_ankle_len = 1
    # back_leg_len = 5
    # back_ankle_len = 5

    # res_model_path = f"res_model/model_{front_leg_len}_{front_ankle_len}_{back_leg_len}_{back_ankle_len}.xml"

    # generate_model(base_model_path, res_model_path, front_leg_len, front_ankle_len, back_leg_len, back_ankle_len)

    # # 5
    # front_leg_len = 1
    # front_ankle_len = 1
    # back_leg_len = 5
    # back_ankle_len = 2

    # res_model_path = f"res_model/model_{front_leg_len}_{front_ankle_len}_{back_leg_len}_{back_ankle_len}.xml"

    # generate_model(base_model_path, res_model_path, front_leg_len, front_ankle_len, back_leg_len, back_ankle_len)

    # # 6
    # front_leg_len = 1
    # front_ankle_len = 1
    # back_leg_len = 2
    # back_ankle_len = 5

    # res_model_path = f"res_model/model_{front_leg_len}_{front_ankle_len}_{back_leg_len}_{back_ankle_len}.xml"

    # generate_model(base_model_path, res_model_path, front_leg_len, front_ankle_len, back_leg_len, back_ankle_len)

    # # 7
    # front_leg_len = 1
    # front_ankle_len = 15
    # back_leg_len = 1
    # back_ankle_len = 15

    # res_model_path = f"res_model/model_{front_leg_len}_{front_ankle_len}_{back_leg_len}_{back_ankle_len}.xml"

    # generate_model(base_model_path, res_model_path, front_leg_len, front_ankle_len, back_leg_len, back_ankle_len)

    run_scene("res_model/model_1_1_5_2.xml")

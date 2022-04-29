#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：graduation_experiment 
@File    ：config.py
@Author  ：xxuanZhu
@Date    ：2022/4/23 11:43 
@Purpose :
'''

from agent import Agent
from env import Env

DIC_AGENTS = {
    "myAgent": Agent

}

DIC_ENVS = {
    "cityflow": Env
}

# 环境相关config
DIC_EXPERIMENT_CONFIG = {
    "RUN_COUNTS": 3600,  # 运行counts
    "NUM_ROUNDS": 100,
    "MODEL_NAME": "MyAgent",  # 模型名称

    "NUM_GENERATORS": 4,
    "MODEL_POOL": False,

    "TRAFFIC_FILE": "flow.json",
    "ROADNET_FILE": "roadnet_6_6.json",
}

# agetn相关config
DIC_MYAGENT_AGENT_CONFIG = {

    "EPOCHS": 100,
    "SAMPLE_SIZE": 1000,
    "MAX_MEMORY_LEN": 10000,
    "UPDATE_Q_BAR_FREQ": 5,  # 每多少步更新

    # network相关参数
    "N_LAYER": 2,

}

# 模拟环境config
DIC_TRAFFIC_ENV_CONF = {
    "MIN_ACTION_TIME": 10,
    "YELLOW_TIME": 5,
    "TOP_K_ADJACENCY": 5,
    "PRETRAIN": False,
    "AGGREGATE": False,
    "USE_LANE_ADJACENCY": False,
    "ADJACENCY_BY_CONNECTION_OR_GEO":False,
    "NUM_AGENTS": 5,  # agent数量
    "NUM_INTERSECTIONS": 10,  # intersection的数量
    "ACTION_PATTERN": "set",  # 直接设置，不是切换
    "MEASURE_TIME": 10,
    "TOP_K_ADJACENCY": 5,
    "SIMULATOR_TYPE": "cityflow",
    "MODEL_NAME": "MyAgent",
    "SAVEREPLAY": True,
    "NUM_ROW": 6,
    "NUM_COL": 6,
    "ROADNET_FILE": "roadnet_6_6.json",

    "LIST_STATE_FEATURE": [
        "cur_phase",
        "lane_num_vehicle",
        "adjacency_matrix",
    ],

    "PHASE": {
        "cityflow": {
            # 0: [0, 0, 0, 0, 0, 0, 0, 0],
            # ->东 左转/直行/右转 ->西  ->南  ->北
            1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],  # 'WSES',  东西直行
            2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],  # 'NSSS',  南北直行
            3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],  # 'WLEL',  东西左转
            4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]  # 'NLSL',   南北左转
        },
    }

}

DIC_PATH = {
    "PATH_TO_DATA": "data/",
    "PATH_TO_MODEL": "model/default",
    "PATH_TO_WORK_DIRECTORY": "records/default",
    "PATH_TO_ERROR": "errors/default",
    "PATH_TO_PRETRAIN_MODEL": "model/default",
    "PATH_TO_PRETRAIN_WORK_DIRECTORY": "records/default",
    "PATH_TO_PRETRAIN_DATA": "data",
    "PATH_TO_AGGREGATE_SAMPLES": "records/initial",
    "PATH_TO_ERROR": "errors/default"

}

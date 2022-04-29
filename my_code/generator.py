#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：graduation 
@File    ：generator.py
@Author  ：xxuanZhu
@Date    ：2022/4/28 16:49 
@Purpose :
'''
import copy
import os
from config import DIC_AGENTS, DIC_ENVS

class Generator:
    def __init__(self, cnt_round,
                 cnt_generator,
                 dic_path,
                 dic_exp_conf,
                 dic_agent_conf,
                 dic_traffic_env_conf
                 ):
        self.cnt_round = cnt_round
        self.cnt_generator = cnt_generator
        self.dic_experiment_config = dic_exp_conf
        self.dic_agent_config = copy.deepcopy(dic_agent_conf)
        self.dic_traffic_env_config = dic_traffic_env_conf
        self.dic_save_path = dic_path
        self.agents = [None]*dic_traffic_env_conf['NUM_AGENTS']

        self.path_to_log = os.path.join(self.dic_save_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                    "round_" + str(self.cnt_round), "generator_" + str(self.cnt_generator))
        if not os.path.exists(self.path_to_log):
            os.makedirs(self.path_to_log)

        self.env = DIC_ENVS[dic_traffic_env_conf["SIMULATOR_TYPE"]](
                              path_to_log = self.path_to_log,
                              path_to_work_directory = self.dic_save_path["PATH_TO_WORK_DIRECTORY"],
                              dic_traffic_env_conf = self.dic_traffic_env_config)



    def generate(self):
        print("generator is working!")

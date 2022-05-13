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
import time

from config import DIC_ENVS


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
        self.agents = [None] * dic_traffic_env_conf['NUM_AGENTS']
        # TODO
        self.price_agent = None

        self.path_to_log = os.path.join(self.dic_save_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                        "round_" + str(self.cnt_round), "generator_" + str(self.cnt_generator))
        if not os.path.exists(self.path_to_log):
            os.makedirs(self.path_to_log)

        self.env = DIC_ENVS[dic_traffic_env_conf["SIMULATOR_TYPE"]](
            path_to_log=self.path_to_log,
            path_to_work_directory=self.dic_save_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=self.dic_traffic_env_config)

    # 收费+交通灯逻辑
    def generate(self):
        reset_env_start_time = time.time()
        done = False
        state = self.env.reset()
        step_num = 0
        reset_env_time = time.time() - reset_env_start_time

        while not done and step_num < int(self.dic_experiment_config["RUN_COUNTS"]/self.dic_traffic_env_config["MIN_ACTION_TIME"]):

            price_aciton_list = []
            price_step_start_time = time.time()

            for i in range(self.dic_traffic_env_config["NUM_PRICE_AGENT"]):
                if self.dic_experiment_config["MODEL_NAME"] in ["MyAgent"]:
                    price_state = state
                    # TODO
                    action, _ = self.price_agent.choose_price(step_num, price_state)
                price_aciton_list = action
            # TODO
            next_state, reward, done, _ = self.env.price_step(price_aciton_list)  # 收费作用


        print("generator is working!")

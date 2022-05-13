#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：graduation 
@File    ：pipeline.py
@Author  ：xxuanZhu
@Date    ：2022/4/28 15:07 
@Purpose :
'''
import json
import os
import shutil
import random
import time
from generator import Generator


class Pipeline:

    def __init__(self, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path):

        self.dic_experiment_config = dic_exp_conf
        self.dic_agent_config = dic_agent_conf
        self.dic_traffic_env_config = dic_traffic_env_conf
        self.dic_save_path = dic_path

        self._check_path()
        self._save_config_file()
        if self.dic_traffic_env_config["SIMULATOR_TYPE"] == 'cityflow':
            self._save_cityflow_file()



    def run(self, multi_process=False):
        print("runing~")
        # 保存一些时间
        f_time = open(os.path.join(self.dic_save_path["PATH_TO_WORK_DIRECTORY"], "running_time.csv"), "w")
        f_time.write("generator_time\tmaking_samples_time\tupdate_network_time\ttest_evaluation_times\tall_times\n")
        f_time.close()


        for cnt_round in range(self.dic_experiment_config["NUM_ROUNDS"]):
            print("round %d starts" % cnt_round)
            round_start_time = time.time()

            process_list = []
            print("==============  generator =============")
            generator_start_time = time.time()

            if multi_process:
                pass
            else:
                for cnt_generator in range(self.dic_experiment_config["NUM_GENERATORS"]):
                    self.generator_wrapper(cnt_round=cnt_round,
                                           cnt_generator=cnt_generator,
                                           dic_path=self.dic_save_path,
                                           dic_exp_conf=self.dic_experiment_config,
                                           dic_agent_conf=self.dic_agent_config,
                                           dic_traffic_env_conf=self.dic_traffic_env_config,
                                           )
            generator_end_time = time.time()
            generator_total_time = generator_end_time - generator_start_time

        if self.dic_experiment_config["MODEL_NAME"] in self.dic_experiment_config["LIST_MODEL_NEED_TO_UPDATE"]:
            if multi_process:
                pass
            else:
                # 更新model
                self.updater_wrapper(cnt_round=0,
                                     dic_agent_conf=self.dic_agent_config,
                                     dic_exp_conf=self.dic_experiment_config,
                                     dic_traffic_env_conf=self.dic_traffic_env_config,
                                     dic_path=self.dic_save_path)









    # 检查保存目录是否存在，不存在则进行创建
    def _check_path(self):
        if os.path.exists(self.dic_save_path["PATH_TO_WORK_DIRECTORY"]):
            if self.dic_save_path["PATH_TO_WORK_DIRECTORY"] != "records/default":
                raise FileExistsError
            else:
                pass
        else:
            os.makedirs(self.dic_save_path["PATH_TO_WORK_DIRECTORY"])

        # model存在路径
        if os.path.exists(self.dic_save_path["PATH_TO_MODEL"]):
            if self.dic_save_path["PATH_TO_MODEL"] != "model/default":
                raise FileExistsError
            else:
                pass
        else:
            os.makedirs(self.dic_save_path["PATH_TO_MODEL"])

    # 保存config配置
    def _save_config_file(self, path=None):
        if path == None:
            path = self.dic_save_path["PATH_TO_WORK_DIRECTORY"]
        json.dump(self.dic_experiment_config, open(os.path.join(path, "experiment.config"), "w"), indent=4)

        json.dump(self.dic_agent_config, open(os.path.join(path, "agent.config"), "w"),
                  indent=4)
        json.dump(self.dic_traffic_env_config,
                  open(os.path.join(path, "traffic_env.config"), "w"), indent=4)

    def _save_cityflow_file(self, path=None):
        if path == None:
            path = self.dic_save_path["PATH_TO_WORK_DIRECTORY"]

        shutil.copy(os.path.join(self.dic_save_path["PATH_TO_DATA"], self.dic_experiment_config["TRAFFIC_FILE"]),
                    os.path.join(path, self.dic_experiment_config["TRAFFIC_FILE"]))
        shutil.copy(os.path.join(self.dic_save_path["PATH_TO_DATA"], self.dic_experiment_config["ROADNET_FILE"]),
                    os.path.join(path, self.dic_experiment_config["ROADNET_FILE"]))

    # 与cityflow进行交互
    def generator_wrapper(self, cnt_round, cnt_generator, dic_path, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf):
        generator = Generator(cnt_round=cnt_round,
                              cnt_generator=cnt_generator,
                              dic_path=dic_path,
                              dic_exp_conf=dic_exp_conf,
                              dic_agent_conf=dic_agent_conf,
                              dic_traffic_env_conf=dic_traffic_env_conf
                              )
        print("make generator")
        generator.generate()
        print("generator_wrapper end")
        return

    # 更新model
    def updater_wrapper(self, cnt_round, dic_agent_conf, dic_exp_conf, dic_traffic_env_conf, dic_path):
        pass
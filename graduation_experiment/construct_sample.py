#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：graduation_experiment 
@File    ：construct_sample.py
@Author  ：xxuanZhu
@Date    ：2022/4/24 17:27 
@Purpose :
'''

import os
import numpy as np
import pickle
import traceback
import pandas as pd

class ConstructSample:

    def __init__(self, path_to_samples, cnt_round, dic_traffic_env_conf):
        self.parent_dir = path_to_samples
        self.path_to_samples = path_to_samples + "/round_" + str(cnt_round)
        self.cnt_round = cnt_round
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.logging_data_list_per_gen = None
        self.hidden_states_list = None
        self.samples = []
        self.samples_all_intersection = [None]*self.dic_traffic_env_conf['NUM_INTERSECTIONS']

    # 读取数据
    def load_data(self, folder, i):

        try:
            f_logging_data = open(os.path.join(self.path_to_samples, folder, "inter_{0}.pkl".format(i)), "rb")
            logging_data = pickle.load(f_logging_data)
            f_logging_data.close()
            return 1, logging_data

        except Exception as e:
            print("Error occurs when making samples for inter {0}".format(i))
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            return 0, None

    # 读取所有intersection的数据，为每个intersection单独生成一个data数据文件夹
    def load_data_for_system(self, folder):
        '''
        Load data for all intersections in one folder
        :param folder:
        :return: a list of logging data of one intersection for one folder
        '''
        self.logging_data_list_per_gen = []
        # load settings
        print("Load data for system in ", folder)
        self.measure_time = self.dic_traffic_env_conf["MEASURE_TIME"]
        self.interval = self.dic_traffic_env_conf["MIN_ACTION_TIME"]

        for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
            pass_code, logging_data = self.load_data(folder, i)
            if pass_code == 0:
                return 0
            self.logging_data_list_per_gen.append(logging_data)
        return 1

    # 读取hidden state的数据
    def load_hidden_state_for_system(self, folder):
        print("loading hidden states: {0}".format(os.path.join(self.path_to_samples, folder, "hidden_states.pkl")))
        # load settings
        if self.hidden_states_list is None:
            self.hidden_states_list = []

        try:
            f_hidden_state_data = open(os.path.join(self.path_to_samples, folder, "hidden_states.pkl"), "rb")
            hidden_state_data = pickle.load(f_hidden_state_data) # hidden state_data is a list of numpy array
            # print(hidden_state_data)
            print(len(hidden_state_data))
            hidden_state_data_h_c = np.stack(hidden_state_data, axis=2)
            hidden_state_data_h_c = pd.Series(list(hidden_state_data_h_c))
            next_hidden_state_data_h_c = hidden_state_data_h_c.shift(-1)
            hidden_state_data_h_c_with_next = pd.concat([hidden_state_data_h_c,next_hidden_state_data_h_c], axis=1)
            hidden_state_data_h_c_with_next.columns = ['cur_hidden','next_hidden']
            self.hidden_states_list.append(hidden_state_data_h_c_with_next[:-1].values)
            return 1
        except Exception as e:
            print("Error occurs when loading hidden states in ", folder)
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            return 0


    # 根据feature构建state
    def construct_state(self,features,time,i):
        '''

        :param features:
        :param time:
        :param i:  intersection id
        :return:
        '''

        state = self.logging_data_list_per_gen[i][time]
        assert time == state["time"]
        if self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
            state_after_selection = {}
            for key, value in state["state"].items():
                if key in features:
                    if "cur_phase" in key:
                        state_after_selection[key] = self.dic_traffic_env_conf['PHASE'][self.dic_traffic_env_conf['SIMULATOR_TYPE']][value[0]]
                    else:
                        state_after_selection[key] = value
        else:
            state_after_selection = {key: value for key, value in state["state"].items() if key in features}
        # print(state_after_selection)
        return state_after_selection


    def _construct_state_process(self, features, time, state, i):
        assert time == state["time"]
        if self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
            state_after_selection = {}
            for key, value in state["state"].items():
                if key in features:
                    if "cur_phase" in key:
                        state_after_selection[key] = self.dic_traffic_env_conf['PHASE'][self.dic_traffic_env_conf['SIMULATOR_TYPE']][value[0]]
                    else:
                        state_after_selection[key] = value
        else:
            state_after_selection = {key: value for key, value in state["state"].items() if key in features}
        return state_after_selection, i


    # 获取reward计算需要的一些数据
    def get_reward_from_features(self, rs):
        reward = {}
        reward["sum_lane_queue_length"] = np.sum(rs["lane_queue_length"])
        reward["sum_lane_wait_time"] = np.sum(rs["lane_sum_waiting_time"])
        reward["sum_lane_num_vehicle_left"] = np.sum(rs["lane_num_vehicle_left"])
        reward["sum_duration_vehicle_left"] = np.sum(rs["lane_sum_duration_vehicle_left"])
        reward["sum_num_vehicle_been_stopped_thres01"] = np.sum(rs["lane_num_vehicle_been_stopped_thres01"])
        reward["sum_num_vehicle_been_stopped_thres1"] = np.sum(rs["lane_num_vehicle_been_stopped_thres1"])
        ##TODO pressure
        reward['pressure'] = np.sum(rs["pressure"])
        return reward


    # 计算reward
    def cal_reward(self, rs, rewards_components):
        r = 0
        for component, weight in rewards_components.items():
            if weight == 0:
                continue
            if component not in rs.keys():
                continue
            if rs[component] is None:
                continue
            r += rs[component] * weight
        return r


    # 生成reward
    def construct_reward(self,rewards_components,time, i):

        rs = self.logging_data_list_per_gen[i][time + self.measure_time - 1]
        assert time + self.measure_time - 1 == rs["time"]
        rs = self.get_reward_from_features(rs['state'])
        r_instant = self.cal_reward(rs, rewards_components)

        # average
        list_r = []
        for t in range(time, time + self.measure_time):
            #print("t is ", t)
            rs = self.logging_data_list_per_gen[i][t]
            assert t == rs["time"]
            rs = self.get_reward_from_features(rs['state'])
            r = self.cal_reward(rs, rewards_components)
            list_r.append(r)
        r_average = np.average(list_r)

        return r_instant, r_average

    # 判断action是否合法
    def judge_action(self,time,i):
        if self.logging_data_list_per_gen[i][time]['action'] == -1:
            raise ValueError
        else:
            return self.logging_data_list_per_gen[i][time]['action']


    # 保存相关带reward的数据
    def make_reward(self, folder, i):
        '''
        make reward for one folder and one intersection,
        add the samples of one intersection into the list.samples_all_intersection[i]
        :param i: intersection id
        :return:
        '''
        if self.samples_all_intersection[i] is None:
            self.samples_all_intersection[i] = []

        if i % 100 == 0:
            print("make reward for inter {0} in folder {1}".format(i, folder))

        list_samples = []

        try:
            total_time = int(self.logging_data_list_per_gen[i][-1]['time'] + 1)
            # construct samples
            time_count = 0
            for time in range(0, total_time - self.measure_time + 1, self.interval):
                state = self.construct_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"], time, i)
                reward_instant, reward_average = self.construct_reward(self.dic_traffic_env_conf["DIC_REWARD_INFO"],
                                                                       time, i)
                action = self.judge_action(time, i)

                if time + self.interval == total_time:
                    next_state = self.construct_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"],
                                                      time + self.interval - 1, i)

                else:
                    next_state = self.construct_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"],
                                                      time + self.interval, i)
                sample = [state, action, next_state, reward_average, reward_instant, time,
                          folder+"-"+"round_{0}".format(self.cnt_round)]
                list_samples.append(sample)


            # list_samples = self.evaluate_sample(list_samples)
            self.samples_all_intersection[i].extend(list_samples)
            return 1
        except Exception as e:
            print("Error occurs when making rewards in generator {0} for intersection {1}".format(folder, i))
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            return 0


    def make_reward_for_system(self):
        '''
        Iterate all the generator folders, and load all the logging data for all intersections for that folder
        At last, save all the logging data for that intersection [all the generators]
        :return:
        '''
        for folder in os.listdir(self.path_to_samples):
            print(folder)
            if "generator" not in folder:
                continue

            if not self.evaluate_sample(folder) or not self.load_data_for_system(folder):
                continue

            for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
                pass_code = self.make_reward(folder, i)
                if pass_code == 0:
                    continue

        for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
            self.dump_sample(self.samples_all_intersection[i],"inter_{0}".format(i))


    # 保存hidden层数据
    def dump_hidden_states(self, folder):
        total_hidden_states = np.vstack(self.hidden_states_list)
        print("dump_hidden_states shape:",total_hidden_states.shape)
        if folder == "":
            with open(os.path.join(self.parent_dir, "total_hidden_states.pkl"),"ab+") as f:
                pickle.dump(total_hidden_states, f, -1)
        elif "inter" in folder:
            with open(os.path.join(self.parent_dir, "total_hidden_states_{0}.pkl".format(folder)),"ab+") as f:
                pickle.dump(total_hidden_states, f, -1)
        else:
            with open(os.path.join(self.path_to_samples, folder, "hidden_states_{0}.pkl".format(folder)),'wb') as f:
                pickle.dump(total_hidden_states, f, -1)


    # def evaluate_sample(self,list_samples):
    #     return list_samples


    # 评估sample是否能够去生成reward
    def evaluate_sample(self, generator_folder):
        return True
        print("Evaluate samples")
        list_files = os.listdir(os.path.join(self.path_to_samples, generator_folder, ""))
        df = []
        # print(list_files)
        for file in list_files:
            if ".csv" not in file:
                continue
            data = pd.read_csv(os.path.join(self.path_to_samples, generator_folder, file))
            df.append(data)
        df = pd.concat(df)
        num_vehicles = len(df['Unnamed: 0'].unique()) -len(df[df['leave_time'].isna()]['leave_time'])
        if num_vehicles < self.dic_traffic_env_conf['VOLUME']* self.dic_traffic_env_conf['NUM_ROW'] and self.cnt_round > 40: # Todo Heuristic
            print("Dumpping samples from ",generator_folder)
            return False
        else:
            return True


    # 把sample进行记录
    def dump_sample(self, samples, folder):
        if folder == "":
            with open(os.path.join(self.parent_dir, "total_samples.pkl"),"ab+") as f:
                pickle.dump(samples, f, -1)
        elif "inter" in folder:
            with open(os.path.join(self.parent_dir, "total_samples_{0}.pkl".format(folder)),"ab+") as f:
                pickle.dump(samples, f, -1)
        else:
            with open(os.path.join(self.path_to_samples, folder, "samples_{0}.pkl".format(folder)),'wb') as f:
                pickle.dump(samples, f, -1)
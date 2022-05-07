#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：graduation 
@File    ：env.py
@Author  ：xxuanZhu
@Date    ：2022/4/29 15:07 
@Purpose :
'''
import json
import os.path
import time
from copy import deepcopy

import jpype
import numpy as np
import pandas as pd

import config


def _get_top_k_lane(lane_id_list, top_k_input):
    top_k_lane_indexes = []
    for i in range(top_k_input):
        lane_id = lane_id_list[i] if i < len(lane_id_list) else None
        top_k_lane_indexes.append(lane_id)
    return top_k_lane_indexes


class Intersection:
    def __init__(self, intersection_id, dic_traffic_env_config, engine, light_id_dict, path_to_log):
        self.intersection_id = intersection_id

        self.intersection_name = "intersection_{0}_{1}".format(intersection_id[0], intersection_id[1])

        self.engine = engine
        self.dic_traffic_env_config = dic_traffic_env_config

        self.fast_compute = dic_traffic_env_config['FAST_COMPUTE']

        self.controlled_model = dic_traffic_env_config['MODEL_NAME']
        self.path_to_log = path_to_log

        self.list_approaches = ["W", "E", "N", "S"]
        self.dic_approach_to_node = {"E": 0, "N": 1, "W": 2, "S": 3}

        self.dic_entering_approach_to_edge = {"W": "road_{0}_{1}_0".format(intersection_id[0] - 1, intersection_id[1])}
        self.dic_entering_approach_to_edge.update({"E": "road_{0}_{1}_2".format(intersection_id[0] + 1, intersection_id[1])})
        self.dic_entering_approach_to_edge.update({"S": "road_{0}_{1}_1".format(intersection_id[0], intersection_id[1] - 1)})
        self.dic_entering_approach_to_edge.update({"N": "road_{0}_{1}_3".format(intersection_id[0], intersection_id[1] + 1)})

        self.dic_exiting_approach_to_edge = {
            approach: "road_{0}_{1}_{2}".format(intersection_id[0], intersection_id[1], self.dic_approach_to_node[approach]) for
            approach in self.list_approaches}
        self.dic_entering_approach_lanes = {"W": [0], "E": [0], "S": [0], "N": [0]}
        self.dic_exiting_approach_lanes = {"W": [0], "E": [0], "S": [0], "N": [0]}


        self.lane_length = 300
        self.terminal_length  = 50
        self.grid_length = 5
        self.num_grid = int(self.lane_length // self.grid_length)

        self.list_phases = dic_traffic_env_config["PHASE"][dic_traffic_env_config["SIMULATOR_TYPE"]]


        self.list_entering_lanes = []
        for approach in self.list_approaches:
            self.list_entering_lanes += [self.dic_entering_approach_to_edge[approach] + "_" + str(i) for i in
                                         range(sum(list(dic_traffic_env_config["LANE_NUM"].values())))]
        self.list_exiting_lanes = []
        for approach in self.list_approaches:
            self.list_exiting_lanes += [self.dic_exiting_approach_to_edge[approach] + '_' + str(i) for i in
                                        range(sum(list(dic_traffic_env_config["LANE_NUM"].values())))]

        self.list_lanes = self.list_entering_lanes + self.list_exiting_lanes

        self.adjacency_row = light_id_dict["adjacency_row"]
        self.neighbor_ENWS = light_id_dict['neighbor_ENWS']
        self.neighbor_lanes_ENWS = light_id_dict['entering_lane_ENWS']

        self._adjacency_row_lanes = {}

        for lane_id in self.list_entering_lanes:
            if lane_id in light_id_dict["adjacency_matrix_lane"]:
                self._adjacency_row_lanes[lane_id] = light_id_dict["adjacency_matrix_lane"][lane_id]
            else:
                self._adjacency_row_lanes[lane_id] = [
                    _get_top_k_lane([], self.dic_traffic_env_config["TOP_K_ADJACENCY_LANE"]),
                    _get_top_k_lane([], self.dic_traffic_env_config["TOP_K_ADJACENCY_LANE"])]

        self.adjacency_row_lane_id_local = {}
        for index, lane_id in enumerate(self.list_entering_lanes):
            self.adjacency_row_lane_id_local[lane_id] = index

        self.dic_lane_vehicle_previous_step = {}
        self.dic_lane_waiting_vehicle_count_previous_step = {}
        self.dic_vehicle_speed_previous_step = {}
        self.dic_vehicle_distance_previous_step = {}

        self.dic_lane_vehicle_current_step = {}
        self.dic_lane_waiting_vehicle_count_current_step = {}
        self.dic_vehicle_speed_current_step = {}
        self.dic_vehicle_distance_current_step = {}

        self.list_lane_vehicle_previous_step = []
        self.list_lane_vehicle_current_step = []

        # -1: all yellow, -2: all red, -3: none
        self.all_yellow_phase_index = -1
        self.all_red_phase_index = -2

        self.current_phase_index = 1
        self.previous_phase_index = 1

        self.engine.set_tl_phase(self.intersection_name, self.current_phase_index)


        path_to_log_file = os.path.join(self.path_to_log, "signal_inter_{0}.txt".format(self.intersection_name))
        df = [self.get_current_time(), self.current_phase_index]
        df = pd.DataFrame(df)
        df = df.transpose()
        df.to_csv(path_to_log_file, mode='a', header=False, index=False)

        self.next_phase_to_set_index = None
        self.current_phase_duration = -1
        self.all_red_flag = False
        self.all_yellow_flag = False
        self.flicker = 0

        self.dic_vehicle_min_speed = {}  # this second
        self.dic_vehicle_arrive_leave_time = dict()  # cumulative

        self.dic_feature = {}  # this second
        self.dic_feature_previous_step = {}  # this second

    # 生成top k 进入的lane 和离开的lane
    def build_adjacency_row_lane(self, lane_id_to_global_index_dict):
        self.adjacency_row_lanes = []
        for entering_lane_id in self.list_entering_lanes:
            _top_k_entering_lane, _top_k_leaving_lane = self._adjacency_row_lanes[entering_lane_id]
            top_k_entering_lane = []
            top_k_leaving_lane = []

            for lane_id in _top_k_entering_lane:
                top_k_entering_lane.append(lane_id_to_global_index_dict[lane_id] if lane_id is not None else -1)

            for lane_id in _top_k_leaving_lane:
                top_k_leaving_lane.append(lane_id_to_global_index_dict[lane_id]
                                          if (lane_id is not None) and (
                        lane_id in lane_id_to_global_index_dict.keys()) \
                                          else -1)
            self.adjacency_row_lanes.append([top_k_entering_lane, top_k_leaving_lane])


    def get_current_time(self):
        return self.engine.get_current_time()

    def update_current_measurements_map(self, simulator_state):

        # 把dict转成list
        def _change_lane_vehicle_dic_to_list(dic_lane_vehicle):
            list_lane_vehicle = []

            for value in dic_lane_vehicle.value():
                list_lane_vehicle.extend(value)

            return list_lane_vehicle

        # 记录当前phase的持续时间
        if self.current_phase_index == self.previous_phase_index:
            self.current_phase_duration += 1
        else:
            self.current_phase_duration = 1

        self.dic_lane_vehicle_current_step = {}
        self.dic_lane_waiting_vehicle_count_current_step = {}
        for lane in self.list_entering_lanes:
            self.dic_lane_vehicle_current_step[lane] = simulator_state["get_lane_vehicles"][lane]
            self.dic_lane_waiting_vehicle_count_current_step[lane] = simulator_state["get_lane_waiting_vehicle_count"][
                lane]
        for lane in self.list_exiting_lanes:
            self.dic_lane_vehicle_current_step[lane] = simulator_state["get_lane_vehicles"][lane]
            self.dic_lane_waiting_vehicle_count_current_step[lane] = simulator_state["get_lane_waiting_vehicle_count"][
                lane]

        self.dic_vehicle_speed_current_step = simulator_state['get_vehicle_speed']
        self.dic_vehicle_distance_current_step = simulator_state['get_vehicle_distance']

        # dict转为list
        self.list_lane_vehicle_current_step = _change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_current_step)
        self.list_lane_vehicle_previous_step = _change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_previous_step)

        list_vehicle_new_arrive = list(
            set(self.list_lane_vehicle_current_step) - set(self.list_lane_vehicle_previous_step))
        # TODO
        list_vehicle_new_left_entering_lane_by_lane = self._update_leave_entering_approach_vehicle()
        list_vehicle_new_left_entering_lane = []
        for l in list_vehicle_new_left_entering_lane_by_lane:
            list_vehicle_new_left_entering_lane += l

        # TODO
        # update vehicle arrive and left time
        self._update_arrive_time(list_vehicle_new_arrive)
        self._update_left_time(list_vehicle_new_left_entering_lane)



        # update feature
        self._update_feature_map(simulator_state)

    def _update_leave_entering_approach_vehicle(self):
        list_entering_lane_vehicle_left = []

        if not self.dic_lane_vehicle_previous_step:
            for lane in self.list_entering_lanes:
                list_entering_lane_vehicle_left.append([])
        else:
            last_step_vehicle_id_list = []
            current_step_vehicle_id_list = []
            for lane in self.list_entering_lanes:
                last_step_vehicle_id_list.extend(self.dic_lane_vehicle_previous_step[lane])  # 在列表末尾追加另一个序列中的值
                current_step_vehicle_id_list.extend(self.dic_lane_vehicle_current_step[lane])
            list_entering_lane_vehicle_left.append(
                list(set(last_step_vehicle_id_list) - set(current_step_vehicle_id_list))
            )
        return list_entering_lane_vehicle_left


class Env:
    def __init__(self, path_to_log, path_to_work_directory, dic_traffic_env_conf):

        self.path_to_log = path_to_log
        self.path_to_work_directory = path_to_work_directory
        self.dic_traffic_env_config = dic_traffic_env_conf
        self.simulator_type = self.dic_traffic_env_config["SIMULATOR_TYPE"]

        self.list_intersection = None
        self.list_inter_log = None
        self.list_lanes = None
        self.system_states = None

        # 检查最小的action的持续时间是否小于黄灯时间
        if self.dic_traffic_env_config["MIN_ACTION_TIME"] <= self.dic_traffic_env_config["YELLOW_TIME"]:
            print("Min action time should include yellow time")

        # 对每个intersection，创建pkl文件，如果存在则从头开始编辑，相当于删除
        for intersection_id in range(self.dic_traffic_env_config["NUM_INTERSECTIONS"]):
            path_to_log_file = os.path.join(self.path_to_log, "intersection_{0}.pkl".format(intersection_id))
            f = open(path_to_log_file, "wb")
            f.close()

    def reset(self):
        print("env reset")
        if not jpype.isJVMStarted():
            jpype.startJVM(classpath=['E:\graduation\lib\*', 'E:\graduation\SEUCityflow-1.0.0.jar'],
                           convertStrings=True)  # jvmargs 根据实际数据设定
        Engine = jpype.JClass('engine')
        self.engine = Engine("E:\graduation\cityflowConfigFile.json", 1)

        if self.dic_traffic_env_config["USE_LANE_ADJACENCY"]:
            self.traffic_light_node_dict = self._adjacency_extraction_lane()
        else:
            self.traffic_light_node_dict = self._adjacency_extraction()

        self.list_intersection = [Intersection((i + 1, j + 1), self.dic_traffic_env_config, self.engine,
                                               self.traffic_light_node_dict[
                                                   "intersection_{0}_{1}".format(i + 1, j + 1)], self.path_to_log)
                                  for i in range(self.dic_traffic_env_config["NUM_ROW"])
                                  for j in range(self.dic_traffic_env_config["NUM_COL"])]

        self.list_inter_log = [[] for i in range(self.dic_traffic_env_config["NUM_ROW"] *
                                                 self.dic_traffic_env_config["NUM_COL"])]

        # 保存每个intersection的index
        self.id_to_index = {}
        count_intersection = 0
        for i in range(self.dic_traffic_env_config["NUM_ROW"]):
            for j in range(self.dic_traffic_env_config["NUM_COL"]):
                self.id_to_index["intersection_{0}_{1}".format(i + 1, j + 1)] = count_intersection
                count_intersection += 1

        self.lane_id_to_index = {}
        count_lane = 0
        for i in range(len(self.list_intersection)):
            for j in range(len(self.list_intersection[i].list_entering_lanes)):
                lane_id = self.list_intersection[i].list_entering_lanes[j]
                if lane_id not in self.lane_id_to_index.keys():
                    self.lane_id_to_index[lane_id] = count_lane
                    count_lane += 1

        for intersection in self.list_intersection:
            intersection.build_adjacency_row_lane(self.lane_id_to_index)

        # 得到系统状态
        system_state_start_time = time.time()
        if self.dic_traffic_env_config["FAST_COMPUTE"]:
            self.system_states = {"get_lane_vehicles": self.engine.get_lane_vehicles(),
                                  "get_lane_waiting_vehicle_count": self.engine.get_lane_waiting_vehicle_count(),
                                  "get_vehicle_speed": None,
                                  "get_vehicle_distance": None
                                  }
        else:
            self.system_states = {"get_lane_vehicles": self.engine.get_lane_vehicles(),
                                  "get_lane_waiting_vehicle_count": self.engine.get_lane_waiting_vehicle_count(),
                                  "get_vehicle_speed": self.engine.get_vehicle_speed(),
                                  "get_vehicle_distance": self.engine.get_vehicle_distance()
                                  }
        print("Get system state time: ", time.time() - system_state_start_time)

        update_start_time = time.time()

        for intersection in self.list_intersection:
            # TODO
            intersection.update_current_measurements_map(self.system_states)
        print("Update_current_measurements_map time: ", time.time() - update_start_time)

        neighbor_start_time = time.time()

        if self.dic_traffic_env_config["NEIGHBOR"]:
            for intersection in self.list_intersection:
                # TODO
                neighbor_intersection_ids = intersection.neighbor_ENWS
                neighbor_intersections = []
                for neighbor_intersection_id in neighbor_intersection_ids:
                    if neighbor_intersection_id is not None:
                        neighbor_intersections.append(
                            self.list_intersection[self.id_to_index[neighbor_intersection_id]])
                    else:
                        # TODO
                        neighbor_intersections.append(None)
                intersection.dic_feature = intersection.update_neighbor_info(neighbor_intersections,
                                                                             deepcopy(intersection.dic_feature))
        print("Update_neighbor time: ", time.time() - neighbor_start_time)

        state, done = self.get_state()

        return state

    def step(self, action):
        step_start_time = time.time()
        list_action_in_sec = [action]
        list_action_in_sec_display = [action]

        for i in range(self.dic_traffic_env_config["MIN_ACTION_TIME"] - 1):
            if self.dic_traffic_env_config["ACTION_PATTERN"] == "switch":
                list_action_in_sec.append(np.zeros_like(action).tolist())
            elif self.dic_traffic_env_config["ACTION_PATTERN"] == "set":
                list_action_in_sec.append(np.copy(action).tolist())
            list_action_in_sec_display.append(np.full_like(action, fill_value=-1).tolist())

        average_reward_action_list = [0] * len(action)
        for i in range(self.dic_traffic_env_config["MIN_ACTION_TIME"]):
            action_in_sec = list_action_in_sec[i]
            action_in_sec_display = list_action_in_sec_display[i]

            instant_time = self.get_current_time()
            self.current_time = self.get_current_time()

            before_action_feature = self.get_feature()

            if self.dic_traffic_env_config['DEBUG']:
                print("time: {0}".format(instant_time))
            else:
                if i == 0:
                    print("time: {0}".format(instant_time))

            self._inner_step(action_in_sec)

            if self.dic_traffic_env_config["DEBUG"]:
                start_time = time.time()

            reward = self.get_reward()

            if self.dic_traffic_env_config["DEBUG"]:
                print("Reward time: {}".format(time.time() - start_time))

            for j in range(len(reward)):
                average_reward_action_list[j] = (average_reward_action_list[j] * i + reward[j]) / (i + 1)

            # log
            self.log(cur_time=instant_time, before_action_feature=before_action_feature,
                     action=action_in_sec_display)

            # 获得下个状态
            next_state, done = self.get_state()

        print("Step time: ", time.time() - step_start_time)
        return next_state, reward, done, average_reward_action_list

    def _adjacency_extraction(self):
        traffic_light_node_dict = {}
        file = os.path.join(self.path_to_work_directory, self.dic_traffic_env_config["ROADNET_FILE"])
        with open("{0}".format(file)) as json_data:
            net = json.load(json_data)
            for intersection in net['intersections']:
                if not intersection['virtual']:
                    traffic_light_node_dict[intersection['id']] = {'location': {'x': float(intersection['point']['x']),
                                                                                'y': float(intersection['point']['y'])},
                                                                   "total_intersection_num": None,
                                                                   "adjacency_row": None,
                                                                   "intersection_id_to_index": None,
                                                                   "neighbor_ENWS": None,
                                                                   "entering_lane_ENWS": None}
            top_k = self.dic_traffic_env_config["TOP_K_ADJACENCY"]
            total_intersection_num = len(traffic_light_node_dict.keys())
            intersection_id_to_index = {}

            edge_id_dict = {}
            for road in net['roads']:
                if road['id'] not in edge_id_dict.keys():
                    edge_id_dict[road['id']] = {}
                edge_id_dict[road['id']]['from'] = road['startIntersection']
                edge_id_dict[road['id']]['to'] = road['endIntersection']
                edge_id_dict[road['id']]['num_of_lane'] = len(road['lanes'])
                edge_id_dict[road['id']]['length'] = np.sqrt(np.square(pd.DataFrame(road['points'])).sum(axis=1)).sum()

            index = 0
            for i in traffic_light_node_dict.keys():
                intersection_id_to_index[i] = index
                index += 1
            # 找每个intersection的相邻intersection, 进入该intersection的lanes
            for i in traffic_light_node_dict.keys():
                traffic_light_node_dict[i]['intersection_id_to_index'] = intersection_id_to_index
                traffic_light_node_dict[i]['neighbor_ENWS'] = []
                traffic_light_node_dict[i]['entering_lane_ENWS'] = {"lane_ids": [], "lane_length": []}
                for j in range(4):
                    road_id = i.replace("intersection", "road") + "_" + str(j)
                    neighboring_intersection = edge_id_dict[road_id]['to']
                    if neighboring_intersection not in traffic_light_node_dict.keys():
                        traffic_light_node_dict[i]['neighbor_ENWS'].append(None)
                    else:
                        traffic_light_node_dict[i]['neighbor_ENWS'].append(neighboring_intersection)

                    # 找进入intersection的lanes
                    for key, value in edge_id_dict.items():
                        if value['from'] == neighboring_intersection and value['to'] == i:
                            neighboring_road = key
                            neighboring_lanes = []
                            for k in range(value['num_of_lane']):
                                neighboring_lanes.append(neighboring_road + "_{0}".format(k))

                            traffic_light_node_dict[i]['entering_lane_ENWS']['lane_ids'].append(neighboring_lanes)
                            traffic_light_node_dict[i]['entering_lane_ENWS']['lane_length'].append(value['length'])
            for i in traffic_light_node_dict.keys():
                location_1 = traffic_light_node_dict[i]['location']

                if not self.dic_traffic_env_config["ADJACENCY_BY_CONNECTION_OR_GEO"]:
                    row = np.array([0] * total_intersection_num)
                    for j in traffic_light_node_dict.keys():
                        location_2 = traffic_light_node_dict[j]['location']
                        dist = Env._cal_distance(location_1, location_2)
                        row[intersection_id_to_index[j]] = dist
                    if len(row) == top_k:
                        adjacency_row_unsorted = np.argpartition(row, -1)[:top_k].tolist()  # 排序，返回topk
                    elif len(row) > top_k:
                        adjacency_row_unsorted = np.argpartition(row, top_k)[:top_k].tolist()
                    else:
                        adjacency_row_unsorted = [k for k in range(total_intersection_num)]
                    adjacency_row_unsorted.remove(intersection_id_to_index[i])
                    traffic_light_node_dict[i]['adjacency_row'] = [intersection_id_to_index[i]] + adjacency_row_unsorted
            else:
                traffic_light_node_dict[i]['adjacency_row'] = [intersection_id_to_index[i]]
                for j in traffic_light_node_dict[i]['neighbor_ENWS']:
                    if j is not None:
                        traffic_light_node_dict[i]['adjacency_row'].append(intersection_id_to_index[j])
                    else:
                        traffic_light_node_dict[i]['adjacency_row'].append(-1)
            traffic_light_node_dict[i]['total_inter_num'] = total_intersection_num

        return traffic_light_node_dict

    # 计算两个点之间的距离
    @staticmethod
    def _cal_distance(location_1, location_2):
        a = np.array(location_1['x'], location_1['y'])
        b = np.array(location_2['x'], location_2['y'])
        return np.sqrt(np.sum((a - b) ** 2))

    def _adjacency_extraction_lane(self):
        pass

    def get_current_time(self):
        return self.engine.get_current_tiem()

    # 把每个inter的feature进行包装，包装成system feature list
    def get_feature(self):
        list_features = [inter.get_feature() for inter in self.list_intersection]
        return list_features

    def get_reward(self):
        list_reward = [inter.get_reward(self.dic_traffic_env_config["DIC_REWARD_INFO"]) for inter in
                       self.list_intersection]
        return list_reward

    def log(self, cur_time, before_action_feature, action):
        for inter_index in range(len(self.list_intersection)):
            self.list_inter_log[inter_index].append({"time": cur_time,
                                                     "state": before_action_feature[inter_index],
                                                     "action": action[inter_index]})

    # 把每个inter的state进行包装，包装成system state list
    def get_state(self):
        # consider neighbor info
        list_state = [inter.get_state(self.dic_traffic_env_config["LIST_STATE_FEATURE"]) for inter in
                      self.list_intersection]
        # 检查一个episode是否结束
        done = self._check_episode_done(list_state)

        # print(list_state)

        return list_state, done

    def _check_episode_done(self, list_state):
        # TODO
        return False

    # 具体动作执行过程
    def _inner_step(self, action):
        for intersection in self.list_intersection:
            # TODO
            intersection.update_previous_measurements()

        for inter_index, intersection in enumerate(self.list_intersection):
            intersection.set_signal(
                action=action[inter_index],
                action_pattern=self.dic_traffic_env_config["ACTION_PATTERN"],
                yellow_time=self.dic_traffic_env_config["YELLOW_TIME"],
                all_red_time=self.dic_traffic_env_config["ALL_RED_TIME"]
            )

        for i in range(int(1 / self.dic_traffic_env_config["INTERVAL"])):
            self.engine.next_step()

        if self.dic_traffic_env_config["DEBUG"]:
            start_time = time.time()

        system_state_start_time = time.time()
        if self.dic_traffic_env_config["FAST_COMPUTE"]:
            self.system_states = {"get_lane_vehicles": self.engine.get_lane_vehicles(),
                                  "get_lane_waiting_vehicle_count": self.engine.get_lane_waiting_vehicle_count(),
                                  "get_vehicle_speed": None,
                                  "get_vehicle_distance": None
                                  }
        else:
            self.system_states = {"get_lane_vehicles": self.engine.get_lane_vehicles(),
                                  "get_lane_waiting_vehicle_count": self.engine.get_lane_waiting_vehicle_count(),
                                  "get_vehicle_speed": self.engine.get_vehicle_speed(),
                                  "get_vehicle_distance": self.engine.get_vehicle_distance()
                                  }

        if self.dic_traffic_env_config["DEBUG"]:
            print("Get system state time: {}".format(time.time() - start_time))

        if self.dic_traffic_env_config['DEBUG']:
            start_time = time.time()

        update_start_time = time.time()
        for intersection in self.list_intersection:
            intersection.update_current_measurements_map(self.system_states)

        if self.dic_traffic_env_config["NEIGHBOR"]:
            for intersection in self.list_intersection:
                neighbor_inter_ids = intersection.neighbor_ENWS
                neighbor_inters = []
                for neighbor_inter_id in neighbor_inter_ids:
                    if neighbor_inter_id is not None:
                        neighbor_inters.append(self.list_intersection[self.id_to_index[neighbor_inter_id]])
                    else:
                        neighbor_inters.append(None)
                intersection.dic_feature = intersection.update_neighbor_info(neighbor_inters,
                                                                             deepcopy(intersection.dic_feature))

        if self.dic_traffic_env_config['DEBUG']:
            print("Update measurements time: {}".format(time.time() - start_time))


if __name__ == '__main__':
    path_to_log = os.path.join(config.DIC_PATH["PATH_TO_WORK_DIRECTORY"])
    # "round_" + str(0), "generator_" + str(0))

    env = Env(path_to_log, config.DIC_PATH["PATH_TO_WORK_DIRECTORY"], config.DIC_TRAFFIC_ENV_CONF)
    state = env.reset()
    print("env finish！")

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

import jpype
import numpy as np
import pandas as pd

import config


class Intersection:
    def __init__(self, intersection_id, dic_traffic_env_conf, engine, light_id_dict, path_to_log):
        self.list_entering_lanes = None


class Env:
    def __init__(self, path_to_log, path_to_work_directory, dic_traffic_env_conf):

        self.path_to_log = path_to_log
        self.path_to_work_directory = path_to_work_directory
        self.dic_traffic_env_config = dic_traffic_env_conf
        self.simulator_type = self.dic_traffic_env_conf["SIMULATOR_TYPE"]

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
        self.engine = Engine("E:\graduation\cityflow_config_file.json", 1)

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
                self.id_to_index["intersection_{0}_{1}".format(i+1, j+1)] = count_intersection
                count_intersection += 1

        self.lane_id_to_index = {}
        count_lane = 0
        for i in range(len(self.list_intersection)):
            for j in range(len(self.list_intersection[i].list_entering_lanes)):
                lane_id = self.list_intersection[i].list_entering_lanes[j]
                if lane_id not in self.lane_id_to_index.keys():
                    self.lane_id_to_index[lane_id] = count_lane
                    count_lane += 1

    def _adjacency_extraction(self):
        traffic_light_node_dict = {}
        file = os.path.join(self.path_to_work_directory, self.dic_traffic_env_config["ROADNET_FILE"])
        with open("{0}".format(file)) as json_data:
            net = json.load(json_data)
            for intersection in net['intersection']:
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


if __name__ == '__main__':
    path_to_log = os.path.join(config.DIC_PATH["PATH_TO_WORK_DIRECTORY"], "train_round",
                               "round_" + str(0), "generator_" + str(0))

    env = Env(path_to_log, config.DIC_PATH["PATH_TO_WORK_DIRECTORY"], config.DIC_TRAFFIC_ENV_CONF)
    env.reset()
    print("env finish！")

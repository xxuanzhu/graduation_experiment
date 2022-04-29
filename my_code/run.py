#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：graduation 
@File    ：run.py
@Author  ：xxuanZhu
@Date    ：2022/4/25 17:13 
@Purpose :
'''

# 把roadnet等一些信息进行记录
import argparse
import copy
import os
import time

import config

from pipeline import Pipeline
ENV_PHASE_CANDIDATE = []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_date", type=str, default='0428_afternoon')
    parser.add_argument('--env', type=int, default=1)  # env=1 means  will run CityFlow
    parser.add_argument('--road_net', type=str, default='6_6')  # roadnet规模，统计intersection数量
    parser.add_argument('--workers', type=int, default=1)  # 运行线程数量
    parser.add_argument('--model', type=str, default='MyAgent')

    tt = parser.parse_args()

    # action候选phase
    ENV_PHASE_CANDIDATE = {
        # 0: [0, 0, 0, 0, 0, 0, 0, 0],
        # ->东 左转/直行/右转 ->西  ->南  ->北
        1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],  # 'WSES',  东西直行
        2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],  # 'NSSS',  南北直行
        3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],  # 'WLEL',  东西左转
        4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]  # 'NLSL',   南北左转
    }

    print('agent_name:%s', tt.model)
    print('ANON_PHASE_REPRE:', ENV_PHASE_CANDIDATE)
    return parser.parse_args()


# 判断多线程是否都在工作，返回不工作的process index，否则返回-1
def check_all_workers_working(list_cur_processers):
    for i in range(len(list_cur_processers)):
        if not list_cur_processers[i].is_alive():
            return i

    return -1


# 把两个字典合并，把dic_to_change添加到dic_tmp
def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)

    return dic_result


# 与env交互一些samples，


# 进行训练，得出phrase


def pipeline_wrapper(dic_experiment_config, dic_agent_config, dic_traffic_env_config, dic_save_path):
    ppl = Pipeline(dic_exp_conf=dic_experiment_config,  # experiment config
                   dic_agent_conf=dic_agent_config,  # RL agent config
                   dic_traffic_env_conf=dic_traffic_env_config,  # the simulation configuration
                   dic_path=dic_save_path)  # the path to save some thing
    ppl.run(multi_process=False)
    print("pipeline_wrapper end")
    return


def main(run_date, env, road_net, model, workers):
    NUM_COL = int(road_net.split('_')[0])
    NUM_ROW = int(road_net.split('_')[1])
    num_intersections = NUM_ROW * NUM_COL
    print('num_intersections:', num_intersections)

    ENVIRONMENT = ["sumo", "cityflow"][env]

    process_list = []
    n_workers = workers
    multi_process = False  # 是否多线程

    dic_experiment_config_extra = {
        "RUN_COUNTS": 3600,  # 运行counts
        "NUM_ROUNDS": 100,
        "MODEL_NAME": model,  # 模型名称

        "NUM_GENERATORS": 4,
        "MODEL_POOL": False,

        "TRAFFIC_FILE": "flow.json",
        "ROADNET_FILE": "roadnet_{0}.json".format(road_net),

    }

    dic_agent_config_extra = {
        "EPOCHS": 100,
        "SAMPLE_SIZE": 1000,
        "MAX_MEMORY_LEN": 10000,
        "UPDATE_Q_BAR_FREQ": 5,  # 每多少步更新

        # network相关参数
        "N_LAYER": 2,
    }

    dic_traffic_env_config_extra = {
        "USE_LANE_ADJACENCY": False,
        "NUM_AGENTS": 5,  # agent数量
        "NUM_INTERSECTIONS": 100,  # intersection的数量
        "ACTION_PATTERN": "set",  # 直接设置，不是切换
        "MEASURE_TIME": 10,
        "TOP_K_ADJACENCY": 5,
        "SIMULATOR_TYPE": "cityflow",
        "MODEL_NAME": "MyAgent",
        "SAVEREPLAY": True,
        "NUM_ROW": NUM_ROW,
        "NUM_COL": NUM_COL,
        "ROADNET_FILE": "roadnet_{0}.json".format(road_net),

        "LIST_STATE_FEATURE": [
            "cur_phase",
            "lane_num_vehicle",
            "adjacency_matrix",
        ],

        "PHASE": {
            "cityflow": ENV_PHASE_CANDIDATE,
        }

    }

    dic_save_path_extra = {
        "PATH_TO_MODEL": os.path.join("model", run_date, "_" + time.strftime('%m_%d_%H_%M_%S',
                                                                             time.localtime(time.time()))),
        "PATH_TO_WORK_DIRECTORY": os.path.join("records", run_date, "_" + time.strftime('%m_%d_%H_%M_%S',
                                                                                        time.localtime(time.time()))),
        "PATH_TO_ERROR": os.path.join("errors", run_date)
    }

    deploy_dic_experiment_config = merge(config.DIC_EXPERIMENT_CONFIG, dic_experiment_config_extra)
    deploy_dic_agent_config = merge(getattr(config, "DIC_{0}_AGENT_CONFIG".format(model.upper())),
                                    dic_agent_config_extra)
    deploy_dic_traffic_env_config = merge(config.DIC_TRAFFIC_ENV_CONF, dic_traffic_env_config_extra)
    deploy_dic_save_path = merge(config.DIC_PATH, dic_save_path_extra)

    if multi_process:
        pass
    else:
        pipeline_wrapper(dic_experiment_config=deploy_dic_experiment_config,
                         dic_agent_config=deploy_dic_agent_config,
                         dic_traffic_env_config=deploy_dic_traffic_env_config,
                         dic_save_path=deploy_dic_save_path)


    if multi_process:
        pass
    else:
        print("start_traffic")

    return run_date



if __name__ == '__main__':
    args = parse_args()

    main(args.run_date, args.env, args.road_net, args.model, args.workers)

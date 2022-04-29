#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：graduation_experiment 
@File    ：agent.py
@Author  ：xxuanZhu
@Date    ：2022/4/23 10:06 
@Purpose :
'''

class Agent(object):
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id="0"):
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.intersection_id = intersection_id


    def choose_action(self):

        raise NotImplementedError


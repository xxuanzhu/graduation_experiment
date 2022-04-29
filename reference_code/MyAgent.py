#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：graduation_experiment 
@File    ：MyAgent.py
@Author  ：xxuanZhu
@Date    ：2022/4/24 17:41 
@Purpose :
'''

from agent import Agent


class MyAgent(Agent):

    def __init__(self,
                 dic_agent_conf=None,
                 dic_traffic_env_conf=None,
                 dic_path=None,
                 cnt_round=None,
                 best_round=None, bar_round=None, intersection_id="0"):
        """
               #1. compute the (dynamic) static Adjacency matrix, compute for each state
               -2. #neighbors: 5 (1 itself + W,E,S,N directions)
               -3. compute len_features
               -4. self.num_actions
               """
        super(MyAgent, self).__init__(
            dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id)

        # TODO
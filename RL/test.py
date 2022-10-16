# -*- coding: utf-8 -*-
# @File    : test.py
# @Date    : 2022-05-18
# @Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来-
# @From    :
import os
import pickle
from pathlib import Path
from collections import namedtuple
import random
from collections.abc import Iterable
import time
from itertools import tee

import networkx
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import networkx as nx
from networkx.algorithms.approximation import steiner_tree
import matplotlib.pyplot as plt

from log import MyLog
from rl import DQN
from env import MulticastEnv
from net import *
from config import Config
from env_config import *

np.random.seed(2022)
torch.random.manual_seed(2022)
random.seed(2022)

mylog = MyLog(Path(__file__), filesave=True, consoleprint=True)
logger = mylog.logger
RouteParams = namedtuple("RouteParams", ('bw', 'delay', 'loss'))
plt.style.use("seaborn-whitegrid")


class Train:
    def __init__(self, Conf, Env, RL, Net, name, mode):
        """
            Conf: class, 配置类
            Env: class, 环境类
            RL: class, 强化学习类
            Net: class, 神经网络类
        """
        self.graph = None
        self.nodes_num = None
        self.edges_num = None

        self.state_channel = 4

        self.record_dict = {}  # 记录训练时所有

        self.config = self.set_initial_config(Conf)  # 配置
        self.config.log_params(logger)

        # 1. 设置 env
        self.env = self.set_initial_env(Env)
        # 2. 初始化 图、节点数、边数
        self.set_init_topology()
        # 3. 设置config中的NUM_STATES NUM_ACTIONS
        self.set_num_states_actions()
        # 4. 设置 RL
        self.rl = self.set_initial_rl(RL, Net)

        self.writer = SummaryWriter(f"./runs/{name}") if mode != 'eval' else None

        self.reward_list_idx = ""

    def set_init_topology(self):
        """
            1. 解析xml文件
            2. 设置 self.graph
                 self.nodes_num
                 self.edges_num
        """
        graph, nodes_num, edges_num = self.parse_xml_topology(self.config.xml_topology_path)
        self.graph = graph
        self.nodes_num = nodes_num
        self.edges_num = edges_num

    def set_num_states_actions(self):
        """
            根据解析topo设置config中的状态空间和动作空间，供深度网络使用
        """
        state_space_num = self.state_channel  # input channel
        action_space_num = self.edges_num  # output channel
        self.config.set_num_states_actions(state_space_num, action_space_num)

    def set_initial_env(self, Env):
        """
            Env类初始化
            :param Env: 环境类
            :return : Env的实例
        """
        env = Env(self.graph, self.config.NUMPY_TYPE)
        return env

    def set_initial_rl(self, RL, Net):
        """
            RL类初始化
        :param RL: RL类 如DQN
        :param Net: Net类
        :return : RL的实例
        """
        rl = RL(self.config, Net)
        return rl

    def set_initial_config(self, Config):
        """
            配置初始化
        :param Config: 配置类
        :return: conf实例
        """
        conf = Config()
        return conf

    @staticmethod
    def parse_xml_topology(topology_path):
        """
            parse topology from topology.xml
        :param topology_path: 拓扑的xml文件路径
        :return: topology graph, networkx.Graph()
                 nodes_num,  int
                 edges_num, int
        """
        tree = ET.parse(topology_path)
        root = tree.getroot()
        topo_element = root.find("topology")
        graph = networkx.Graph()
        for child in topo_element.iter():
            # parse nodes
            if child.tag == 'node':
                node_id = int(child.get('id'))
                graph.add_node(node_id)
            # parse link
            elif child.tag == 'link':
                from_node = int(child.find('from').get('node'))
                to_node = int(child.find('to').get('node'))
                graph.add_edge(from_node, to_node)

        nodes_num = len(graph.nodes)
        edges_num = len(graph.edges)

        print('nodes: ', nodes_num, '\n', graph.nodes, '\n',
              'edges: ', edges_num, '\n', graph.edges)
        return graph, nodes_num, edges_num

    def pkl_file_path_yield(self, pkl_dir, n: int = 3000, step: int = 1):
        """
            生成保存的pickle文件的路径, 按序号递增的方式生成
            :param pkl_dir: Path, pkl文件的目录
            :param n: 截取
            :param step: 间隔
        """
        a = os.listdir(pkl_dir)
        assert n < len(a), "n should small than len(a)"
        b = sorted(a, key=lambda x: int(x.split('-')[0]))
        for p in b[:n:step]:
            yield pkl_dir / p

    def record_route_params(self, episode, bw, delay, loss):
        """
            将route信息存入字典中 {episode: RouteParams}
        :param episode: 训练的第几代， 作为key
        :param bw: 剩余带宽和
        :param delay: 时延和
        :param loss: 丢包率和
        :return: None
        """
        self.record_dict.setdefault(episode, RouteParams(bw, delay, loss))
        with open("./recorder.pkl", "wb+") as f:
            pickle.dump(self.record_dict, f)

    @staticmethod
    def record_reward(r):
        with open('./reward.pkl', 'wb') as f:
            pickle.dump(r, f)

    def choice_multicast_nodes(self):
        """
            随机选择 源节点和目的节点
            source 至少 1
            destination 至少 2
            所有节点至多 nodes_num - 1
        :return: multicast nodes
        """
        _a = list(self.graph.nodes())
        _size = np.random.randint(3, self.nodes_num)
        # 无放回抽取
        multicast_nodes = np.random.choice(_a, size=_size, replace=False)
        multicast_nodes = list(multicast_nodes)
        start_node = multicast_nodes.pop(0)
        end_nodes = multicast_nodes

        return start_node, end_nodes

    @staticmethod
    def flat_and_combine_state(route, bw, delay, loss):
        """
            将二维矩阵展平成一维, 将多个一维拼接
        :param route: 路由矩阵
        :param bw: 剩余带宽矩阵
        :param delay: 时延矩阵
        :param loss: 丢包率矩阵
        :return: 展平的tensor
        """
        if route is not None:
            flatten_route = torch.flatten(torch.from_numpy(route)).unsqueeze(0)
            flatten_bw = torch.flatten(torch.from_numpy(bw)).unsqueeze(0)
            flatten_delay = torch.flatten(torch.from_numpy(delay)).unsqueeze(0)
            flatten_loss = torch.flatten(torch.from_numpy(loss)).unsqueeze(0)
            combine_state = torch.cat([flatten_route, flatten_bw, flatten_delay, flatten_loss], dim=1)
            return combine_state
        else:
            return None

    def combine_state(self, route):
        """
            组合 多channel
        :param route: 路由矩阵
        :return: 多channel tensor
        """
        if route is not None:
            if self.state_channel == 1:
                combine_state = torch.stack(
                    [torch.from_numpy(route) + torch.from_numpy(self.env.get_branches_matrix())], dim=0)
            elif self.state_channel == 2:
                combine_state = torch.stack(
                    [
                        torch.from_numpy(route),
                        torch.from_numpy(self.env.get_branches_matrix()),
                    ],
                    dim=0)
            elif self.state_channel == 4:
                combine_state = torch.stack(
                    [
                        torch.from_numpy(route) + torch.from_numpy(self.env.get_branches_matrix()),
                        torch.from_numpy(self.env.normal_bw_matrix),
                        torch.from_numpy(self.env.normal_delay_matrix),
                        torch.from_numpy(self.env.normal_loss_matrix)
                    ],
                    dim=0)
            else:
                combine_state = torch.stack(
                    [
                        torch.from_numpy(route),
                        torch.from_numpy(self.env.get_branches_matrix()),
                        torch.from_numpy(self.env.normal_bw_matrix),
                        torch.from_numpy(self.env.normal_delay_matrix),
                        torch.from_numpy(self.env.normal_loss_matrix)
                    ],
                    dim=0)
            return torch.unsqueeze(combine_state, dim=0)
        else:
            return None

    @staticmethod
    def kmb_algorithm(graph, src_node, dst_nodes, weight=None):
        """
            经典KMB算法 组播树

        :param graph: networkx.graph
        :param src_node: 源节点
        :param dst_nodes: 目的节点
        :param weight: 计算的权重
        :return: 返回图形的最小Steiner树的近似值
        """
        terminals = [src_node] + dst_nodes
        st_tree = steiner_tree(graph, terminals, weight)
        return st_tree

    @staticmethod
    def spanning_tree(graph, weight=None):
        """
            生成树算法
        :param graph: networkx.graph
        :param weight: 计算的权重
        :return: iterator 最小生成树
        """
        spanning_tree = nx.algorithms.minimum_spanning_tree(graph, weight)
        return spanning_tree

    def get_tree_params(self, tree, graph):
        """
            计算tree的bw, delay, loss 参数和
        :param tree: 要计算的树
        :param graph: 计算图中的数据
        :return: bw, delay, loss, len
        """
        bw, delay, loss = 0, 0, 0
        if isinstance(tree, nx.Graph):
            edges = tree.edges
        elif isinstance(tree, Iterable):
            edges = tree
        else:
            raise ValueError("tree param error")
        num = 0
        for r in edges:
            bw += graph[r[0]][r[1]]["bw"]
            delay += graph[r[0]][r[1]]["delay"]
            loss += graph[r[0]][r[1]]["loss"]
            num += 1
        bw_mean = self.env.find_end_to_end_max_bw(tree, self.env.start, self.env.ends_constant).mean()
        return bw_mean, delay / num, loss / num, len(tree.edges)

    @staticmethod
    def modify_bw_weight(graph):
        """
            将delay取负，越大表示越小
        :param graph: 图
        :return: weight
        """
        _g = graph.copy()
        for edge in graph.edges:
            _g[edge[0]][edge[1]]['bw'] = 1 / (graph[edge[0]][edge[1]]['bw'] + 1)
        return _g

    def get_kmb_params(self, graph, start_node, end_nodes):
        """
            获得以 bw 为权重的 steiner tree 返回该树的 bw和
            获得以 delay 为权重的 steiner tree 返回该树的 delay和
            获得以 loss 为权重的 steiner tree 返回该树的 loss和
            获得以 hope 为权重的 steiner tree 返回该树的 长度length
        :param graph: 图
        :param start_node: 源节点
        :param end_nodes: 目的节点
        :return: bw, delay, loss, length
        """
        _g = self.modify_bw_weight(graph)
        # kmb算法 计算权重为-bw
        kmb_bw_tree = self.kmb_algorithm(_g, start_node, end_nodes,
                                         weight='bw')
        bw_bw, bw_delay, bw_loss, bw_length = self.get_tree_params(kmb_bw_tree, graph)

        # kmb算法 计算权重为delay
        kmb_delay_tree = self.kmb_algorithm(graph, start_node, end_nodes,
                                            weight='delay')
        delay_bw, delay_delay, delay_loss, delay_length = self.get_tree_params(kmb_delay_tree, graph)

        # kmb算法 计算权重为loss
        kmb_loss_tree = self.kmb_algorithm(graph, start_node, end_nodes,
                                           weight='loss')
        loss_bw, loss_delay, loss_loss, loss_length = self.get_tree_params(kmb_loss_tree, graph)

        # kmb算法 为None
        kmb_hope_tree = self.kmb_algorithm(graph, start_node, end_nodes, weight=None)
        length_bw, length_delay, length_loss, length_length = self.get_tree_params(kmb_hope_tree, graph)

        bw_ = [bw_bw, delay_bw, loss_bw, length_bw]
        delay_ = [bw_delay, delay_delay, loss_delay, length_delay]
        loss_ = [bw_loss, delay_loss, loss_loss, length_loss]
        length_ = [bw_length, delay_length, loss_length, length_length]
        return bw_, delay_, loss_, length_

    def get_spanning_tree_params(self, graph):
        """
            获得以 bw 为权重的 spanning tree 返回该树的 bw和
            获得以 delay 为权重的 spanning tree 返回该树的 delay和
            获得以 loss 为权重的 spanning tree 返回该树的 loss和
            获得以 hope 为权重的 spanning tree 返回该树的 长度length
        :param graph:
        :return:
        """
        _g = self.modify_bw_weight(graph)
        spanning_bw_tree = self.spanning_tree(_g, weight='bw')
        bw, _, _, _ = self.get_tree_params(spanning_bw_tree, graph)
        spanning_delay_tree = self.spanning_tree(graph, weight='delay')
        _, delay, _, _ = self.get_tree_params(spanning_delay_tree, graph)
        spanning_loss_tree = self.spanning_tree(graph, weight='loss')
        _, _, loss, _ = self.get_tree_params(spanning_loss_tree, graph)
        spanning_length_tree = self.spanning_tree(graph, weight=None)
        _, _, _, length = self.get_tree_params(spanning_length_tree, graph)

        return bw, delay, loss, length

    def print_train_info(self, episode, index, reward, link):
        logger.info(f"[{episode}][{index}] reward: {reward}")
        logger.info(f"[{episode}][{index}] tree_nodes: {self.env.tree_nodes}")
        logger.info(f"[{episode}][{index}] route_list: {self.env.route_graph.edges}")
        logger.info(f"[{episode}][{index}] branches: {self.env.branches}")
        logger.info(f"[{episode}][{index}] link: {link}")
        logger.info(f"[{episode}][{index}] step_num: {self.env.step_num}")
        # logger.info(f'[{episode}][{index}]: {self.env.route_matrix}')
        logger.info("=======================================================")

    def update(self):
        """
            状态更新 rl学习
            1. 循环代数, 进行训练
            2. 读取一个graph, 环境reset
            3. while True 直到跑出path

            2022/3/17 修改link方向BUG
        """
        pkl_cut_num = self.config.PKL_CUT_NUM
        pkl_step = self.config.PKL_STEP
        loss_step = 0
        for episode in range(self.config.EPISODES):
            # start_node, end_nodes = self.choice_multicast_nodes()
            start_node = 12
            end_nodes = [2, 4, 11]

            logger.info(f"[{episode}] start_node: {start_node}")
            logger.info(f"[{episode}] end_nodes: {end_nodes}")

            episode_reward = np.array([])
            episode_bw, kmb_bw, spanning_bw = np.array([]), np.array([]), np.array([])
            episode_delay, kmb_delay, spanning_delay = np.array([]), np.array([]), np.array([])
            episode_loss, kmb_loss, spanning_loss = np.array([]), np.array([]), np.array([])
            episode_length, kmb_length, spanning_length = np.array([]), np.array([]), np.array([])
            episode_steps = np.array([])

            for index, pkl_path in enumerate(
                    self.pkl_file_path_yield(self.config.pkl_weight_path, n=pkl_cut_num, step=pkl_step)):

                self.env.read_pickle_and_modify(pkl_path)
                state = self.env.reset(start_node, end_nodes)

                reward_temp = 0
                while True:
                    # 1. 二维
                    combine_state = self.combine_state(state)
                    # 2.1 动作选择
                    action = self.rl.choose_action(combine_state, episode)
                    # 2.2 将动作映射为链路
                    link = self.env.map_action(action)
                    # 3. 环境交互
                    new_state, reward, done, flag = self.env.step(link)
                    # 4. 下一个状态
                    combine_new_state = self.combine_state(new_state)
                    # 5. RL学习
                    self.rl.learn(combine_state, action, reward, combine_new_state)
                    reward_temp += reward
                    if len(self.rl.losses) > 0:
                        self.writer.add_scalar("Optim/Loss", self.rl.losses.pop(0), loss_step)
                        loss_step += 1
                    if done:
                        self.rl.finish_n_steps()

                        if flag == "ALL":
                            bw, delay, loss, length = self.env.get_route_params()  # 获得路径的所以链路bw和,delay和,loss和
                            # 添加到数组中
                            episode_bw = np.append(episode_bw, bw)
                            episode_delay = np.append(episode_delay, delay)
                            episode_loss = np.append(episode_loss, loss)
                            episode_length = np.append(episode_length, length)

                            # kmb 算法
                            bw, delay, loss, length = self.get_kmb_params(self.env.graph, start_node, end_nodes)
                            kmb_bw = np.append(kmb_bw, bw)
                            kmb_delay = np.append(kmb_delay, delay)
                            kmb_loss = np.append(kmb_loss, loss)
                            kmb_length = np.append(kmb_length, length)

                        episode_reward = np.append(episode_reward, reward_temp)
                        episode_steps = np.append(episode_steps, self.env.step_num - 1)
                        self.print_train_info(episode, index, reward, link)
                        break

                    # 6. 状态改变
                    state = new_state

            # self.writer.add_scalars('Episode/reward',
            #                         {"reward": episode_reward.mean(), "reward_max": episode_reward.max(initial=0)},
            #                         episode)
            # self.writer.add_scalars('Episode/steps',
            #                         {"reward": episode_steps.mean(), "reward_max": episode_steps.max(initial=0)},
            #                         episode)
            #
            # self.writer.add_scalars('Episode/bw', {"rl": episode_bw.mean(), "kmb_bw": kmb_bw.mean(),
            #                                        }, episode)
            # self.writer.add_scalars('Episode/delay', {"rl": episode_delay.mean(), "kmb_delay": kmb_delay.mean(),
            #                                           }, episode)
            # self.writer.add_scalars('Episode/loss', {"rl": episode_loss.mean(), "kmb_loss": kmb_loss.mean(),
            #                                          }, episode)
            # self.writer.add_scalars('Episode/length', {"rl": episode_length.mean(), "kmb_length": kmb_length.mean(),
            #                                            }, episode)
            self.writer.add_scalar('Episode/reward', episode_reward.mean(), episode)
            self.writer.add_scalar('Episode/steps', episode_steps.mean(), episode)
            self.writer.add_scalar('Episode/bw', episode_bw.mean(), episode)
            self.writer.add_scalar('Episode/delay', episode_delay.mean(), episode)
            self.writer.add_scalar('Episode/loss', episode_loss.mean(), episode)
            self.writer.add_scalar('Episode/length', episode_length.mean(), episode)
            self.writer.add_scalar("learning_rate", self.rl.optimizer.param_groups[0]['lr'], episode)
            self.rl.scheduler.step()
            self.rl.save_weight()

        logger.debug('train over')

    def set_reward_list_idx(self, idx):
        self.reward_list_idx = idx

    def compare_test(self,
                     weight_file):
        self.rl.change_model_mode('eval')

        pkl_cut_num = self.config.PKL_CUT_NUM
        pkl_step = self.config.PKL_STEP
        start_node = 12
        end_nodes = [2, 4, 11]

        episode_bw, kmb_bw, spanning_bw = [], [], []
        episode_delay, kmb_delay, spanning_delay = [], [], []
        episode_loss, kmb_loss, spanning_loss = [], [], []
        episode_length, kmb_length, spanning_length = [], [], []

        self.rl.load_weight(weight_file, None)
        for index, pkl_path in enumerate(
                self.pkl_file_path_yield(self.config.pkl_weight_path, n=pkl_cut_num, step=pkl_step)):

            self.env.read_pickle_and_modify(pkl_path)
            state = self.env.reset(start_node, end_nodes)

            while True:
                # 1. 二维
                combine_state = self.combine_state(state)
                # 2.1 动作选择
                action = self.rl.choose_max_action(combine_state)
                # 2.2 将动作映射为链路
                link = self.env.map_action(action)
                # 3. 环境交互
                new_state, reward, done, flag = self.env.step(link)
                if done:
                    if flag == "ALL":
                        bw, delay, loss, length = self.env.get_route_params()  # 获得路径的所以链路bw和,delay和,loss和
                        # 添加到数组中
                        episode_bw.append(bw)
                        episode_delay.append(delay)
                        episode_loss.append(loss)
                        episode_length.append(length)

                        # kmb 算法
                        bw, delay, loss, length = self.get_kmb_params(self.env.graph, start_node, end_nodes)
                        kmb_bw.append(bw)
                        kmb_delay.append(delay)
                        kmb_loss.append(loss)
                        kmb_length.append(length)
                    break

                # 6. 状态改变
                state = new_state
        self.plot_compare_figure(episode_bw, kmb_bw, "traffic", "mean bw", "bw")
        self.plot_compare_figure(episode_delay, kmb_delay, "traffic", "mean delay", "delay")
        self.plot_compare_figure(episode_loss, kmb_loss, "traffic", "mean loss", "loss")

        # self.plot_compare_figure_subplots(episode_bw, episode_delay, episode_loss, kmb_bw, kmb_delay, kmb_loss)

    def plot_compare_figure(self, rl_result, kmb_result, x_label, y_label, title):
        width = 0.18
        plt.bar(range(len(kmb_result)), rl_result, width, label='rl')

        kmb_bw = [kmb_result[i][0] for i in range(len(kmb_result))]
        kmb_delay = [kmb_result[i][1] for i in range(len(kmb_result))]
        kmb_loss = [kmb_result[i][2] for i in range(len(kmb_result))]

        plt.bar([x + width for x in range(len(kmb_result))], kmb_bw, width, label='kmb_bw')
        plt.bar([x + 2 * width for x in range(len(kmb_result))], kmb_delay, width, label='kmb_delay')
        plt.bar([x + 3 * width for x in range(len(kmb_result))], kmb_loss, width, label='kmb_loss')

        plt.xticks(range(len(kmb_result)), range(len(kmb_result)), rotation=0, fontsize='small')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        # plt.title(title)
        plt.legend(bbox_to_anchor=(0., 1.0), loc='lower left', ncol=4, )

        _path = Path('./images')
        if _path.exists():
            plt.savefig(_path / f'{title}.png')
        else:
            _path.mkdir(exist_ok=True)
            plt.savefig(_path / f'{title}.png')
        plt.show()

    def plot_compare_figure_subplots(self, rl_bw_result, rl_delay_result, rl_loss_result, kmb_bw_result,
                                     kmb_delay_result, kmb_loss_result):
        width = 0.18
        fig, ax = plt.subplots(1, 3)

        def get_bw_delay_loss(result):
            _bw = [result[i][0] for i in range(len(result))]
            _delay = [result[i][1] for i in range(len(result))]
            _loss = [result[i][2] for i in range(len(result))]
            return _bw, _delay, _loss

        def ax_plot(i, kmb_result, rl_result):
            kmb_bw, kmb_delay, kmb_loss = get_bw_delay_loss(kmb_result)
            ax[i].bar(range(len(rl_result)), rl_result, width, label='rl')
            ax[i].bar([x + width for x in range(len(kmb_bw))], kmb_bw, width, label='kmb_bw')
            ax[i].bar([x + 2 * width for x in range(len(kmb_bw))], kmb_delay, width, label='kmb_delay')
            ax[i].bar([x + 3 * width for x in range(len(kmb_bw))], kmb_loss, width, label='kmb_loss')

        # bw
        ax_plot(0, kmb_bw_result, rl_bw_result)
        ax[0].set_ylabel("mean bw")
        ax_plot(1, kmb_delay_result, rl_delay_result)
        ax[1].set_ylabel("mean delay")
        ax_plot(2, kmb_loss_result, rl_loss_result)
        ax[2].set_ylabel("mean loss")

        plt.xticks(range(len(rl_bw_result)), range(len(rl_bw_result)), rotation=0, fontsize='small')
        # plt.title(title)
        plt.legend(bbox_to_anchor=(0., 1.0), loc='lower left', ncol=4, )

        _path = Path('./images')
        if _path.exists():
            plt.savefig(_path / 'result.png')
        else:
            _path.mkdir(exist_ok=True)
            plt.savefig(_path / f'result.png')
        plt.show()

    def read_data_path(self, data_path="./data"):
        data_path = Path(data_path)
        data_path_dict = {"lr": [], "nsteps": [], "batchsize": [], 'egreedy': [], "gamma": [], "update": [],
                          "rewardslist": [], "tau": []}

        for data_doc in data_path.iterdir():
            if data_doc.match('*_lr_*'):
                data_path_dict['lr'].append(data_doc)
            elif data_doc.match('*_nsteps_*'):
                data_path_dict['nsteps'].append(data_doc)
            elif data_doc.match('*_batchsize_*'):
                data_path_dict['batchsize'].append(data_doc)
            elif data_doc.match('*_egreedy_*'):
                data_path_dict['egreedy'].append(data_doc)
            elif data_doc.match('*_gamma_*'):
                data_path_dict['gamma'].append(data_doc)
            elif data_doc.match('*_updatefrequency_*'):
                data_path_dict['update'].append(data_doc)
            elif data_doc.match('*_rewardslist_*'):
                data_path_dict['rewardslist'].append(data_doc)
            elif data_doc.match('*_tau_*'):
                data_path_dict['tau'].append(data_doc)

        return data_path_dict

    def get_diff_data_from_multi_file(self, data_path_list):
        data_dict = {"bw": [], "delay": [], "loss": [], 'length': [], "final_reward": [], "episode_reward": [],
                     "steps": []}

        for path in data_path_list:
            for child in path.iterdir():
                data = np.load(child)
                if child.match("*bw*"):
                    data_dict['bw'].append(data)
                elif child.match('*delay*'):
                    data_dict['delay'].append(data)
                elif child.match("*loss*"):
                    data_dict['loss'].append(data)
                elif child.match("*length*"):
                    data_dict['length'].append(data)
                elif child.match("*final_reward*"):
                    data_dict['final_reward'].append(data)
                elif child.match("*episode_reward*"):
                    data_dict['episode_reward'].append(data)
                elif child.match("*steps*"):
                    data_dict['steps'].append(data)

        return data_dict

    def get_compare_data(self, data_path="./data"):
        data_path_dict = self.read_data_path(data_path)
        data_npy_dict = {}
        for k in data_path_dict.keys():
            data_npy_dict[k] = self.get_diff_data_from_multi_file(data_path_dict[k])

        return data_npy_dict

    def plot_line_chart(self, data_list, name):
        for data in data_list:
            x = range(len(data))
            plt.plot(x, data)
        plt.legend()
        plt.save('./')


if __name__ == '__main__':
    pt_file_path = "./saved_agents/policy_net-2022-05-10-16-05-33.pt"
    exp_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    _name = f"[{exp_time}]_test"
    train = Train(Config, MulticastEnv, DQN, MyMulticastNet3, name=_name, mode='eval')
    train.compare_test(pt_file_path)

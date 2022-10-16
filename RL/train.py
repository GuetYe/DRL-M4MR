# -*- coding: utf-8 -*-
# @File    : train.py
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
import argparse

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import networkx
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import networkx as nx
from networkx.algorithms.approximation import steiner_tree

from log import MyLog
from rl import DQN
from env import MulticastEnv
from net import *
from config import Config

# plt.rcParams['figure.figsize'] = (19.80, 10.80)
plt.style.use('seaborn-whitegrid')
np.random.seed(1)
torch.random.manual_seed(1)
random.seed(1)

RouteParams = namedtuple("RouteParams", ('bw', 'delay', 'loss'))


class Train:
    def __init__(self, Conf, Env, RL, Net, name, mode='train'):
        """
            Conf: class, 配置类
            Env: class, 环境类
            RL: class, 强化学习类
            Net: class, 神经网络类
        """
        self.name = name.strip()
        self.mylog = MyLog(Path(__file__), filesave=True, consoleprint=True, name=self.name)
        self.logger = self.mylog.logger
        self.graph = None
        self.nodes_num = None
        self.edges_num = None

        self.state_channel = 4

        self.record_dict = {}  # 记录训练时所有

        self.config = self.set_initial_config(Conf)  # 配置
        self.config.log_params(self.logger)

        # 1. 设置 env
        self.env = self.set_initial_env(Env)
        # 2. 初始化 图、节点数、边数
        self.set_init_topology()
        # 3. 设置config中的NUM_STATES NUM_ACTIONS
        self.set_num_states_actions()
        # 4. 设置 RL
        self.rl = self.set_initial_rl(RL, Net)

        self.writer = SummaryWriter(f"./runs/{self.name}") if mode != 'eval' else None

        self.reward_list_idx = ""
        self.data_path = f"./data/{self.name}"
        self.image_path = Path('./images')

        self.image_path.mkdir(exist_ok=True)
        if mode != 'eval':
            Path(self.data_path).mkdir(exist_ok=True, parents=True)
        pass

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

    @staticmethod
    def set_initial_config(Config):
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

    @staticmethod
    def pkl_file_path_yield(pkl_dir, s=10, n: int = 3000, step: int = 1):
        """
            生成保存的pickle文件的路径, 按序号递增的方式生成
            :param pkl_dir: Path, pkl文件的目录
            :param s: 开始index
            :param n: 结束index
            :param step: 间隔
        """
        a = os.listdir(pkl_dir)
        assert n < len(a), "n should small than len(a)"
        # print([x.split('-')[0] for x in a])
        b = sorted(a, key=lambda x: int(x.split('-')[0]))
        for p in b[s:n:step]:
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

    def get_spanning_tree_params(self, graph):
        """
            获得以 bw 为权重的 steiner tree 返回该树的 bw和
            获得以 delay 为权重的 steiner tree 返回该树的 delay和
            获得以 loss 为权重的 steiner tree 返回该树的 loss和
            获得以 hope 为权重的 steiner tree 返回该树的 长度length
        :param graph: 图
        :return: bw, delay, loss, length
        """
        _g = self.modify_bw_weight(graph)
        # kmb算法 计算权重为-bw
        kmb_bw_tree = self.spanning_tree(_g, weight='bw')
        bw_bw, bw_delay, bw_loss, bw_length = self.get_tree_params(kmb_bw_tree, graph)

        # kmb算法 计算权重为delay
        kmb_delay_tree = self.spanning_tree(graph, weight='delay')
        delay_bw, delay_delay, delay_loss, delay_length = self.get_tree_params(kmb_delay_tree, graph)

        # kmb算法 计算权重为loss
        kmb_loss_tree = self.spanning_tree(graph, weight='loss')
        loss_bw, loss_delay, loss_loss, loss_length = self.get_tree_params(kmb_loss_tree, graph)

        # kmb算法 为None
        kmb_hope_tree = self.spanning_tree(graph, weight=None)
        length_bw, length_delay, length_loss, length_length = self.get_tree_params(kmb_hope_tree, graph)

        bw_ = [bw_bw, delay_bw, loss_bw, length_bw]
        delay_ = [bw_delay, delay_delay, loss_delay, length_delay]
        loss_ = [bw_loss, delay_loss, loss_loss, length_loss]
        length_ = [bw_length, delay_length, loss_length, length_length]
        return bw_, delay_, loss_, length_

    def get_kmb_params(self, graph, start_node, end_nodes):
        """
            获得以 bw 为权重的 spanning tree 返回该树的 bw和
            获得以 delay 为权重的 spanning tree 返回该树的 delay和
            获得以 loss 为权重的 spanning tree 返回该树的 loss和
            获得以 hope 为权重的 spanning tree 返回该树的 长度length
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

    def print_train_info(self, episode, index, reward, link):
        self.logger.info(f"[{episode}][{index}] reward: {reward}")
        self.logger.info(f"[{episode}][{index}] tree_nodes: {self.env.tree_nodes}")
        self.logger.info(f"[{episode}][{index}] route_list: {self.env.route_graph.edges}")
        self.logger.info(f"[{episode}][{index}] branches: {self.env.branches}")
        self.logger.info(f"[{episode}][{index}] link: {link}")
        self.logger.info(f"[{episode}][{index}] step_num: {self.env.step_num}")
        self.logger.info(f"[{episode}][{index}] alley_num: {self.env.alley_num}")

        # self.logger.info(f'[{episode}][{index}]: {self.env.route_matrix}')
        self.logger.info("=======================================================")

    def update(self):
        """
            状态更新 rl学习
            1. 循环代数, 进行训练
            2. 读取一个graph, 环境reset
            3. while True 直到跑出path

            2022/3/17 修改link方向BUG
        """
        start_time = time.time()
        pkl_cut_num = self.config.PKL_CUT_NUM
        pkl_start = self.config.PKL_START
        pkl_step = self.config.PKL_STEP
        loss_step = 0
        all_episode_reward, all_episode_final_reward, all_episode_steps, all_episode_bw, all_episode_delay, all_episode_loss, all_episode_length = np.array(
            []), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        start_node = self.config.start_node
        end_nodes = self.config.end_nodes
        episode_reward_temp = 0

        self.logger.info(f"start_node: {start_node}")
        self.logger.info(f"end_nodes: {end_nodes}")

        for episode in range(self.config.EPISODES):
            # start_node, end_nodes = self.choice_multicast_nodes()
            episode_reward, episode_final_reward, episode_bw, episode_delay, episode_loss, episode_length, episode_steps = np.array(
                []), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

            for index, pkl_path in enumerate(
                    self.pkl_file_path_yield(self.config.pkl_weight_path, s=pkl_start, n=pkl_cut_num, step=pkl_step)):

                self.env.read_pickle_and_modify(pkl_path)
                state = self.env.reset(start_node, end_nodes)

                reward_temp = 0
                while True:
                    # if self.env.step_num >= len(self.env.edges):
                    #     break
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
                    self.rl.learn(combine_state, action, reward, combine_new_state, episode)
                    reward_temp += reward
                    if len(self.rl.losses) > 0:
                        self.writer.add_scalar("Optim/Loss", self.rl.losses.pop(0), loss_step)
                        loss_step += 1
                    if done:
                        self.rl.finish_n_steps()

                        if flag == "ALL":
                            bw, delay, loss, length, _ = self.env.get_route_params()  # 获得路径的所以链路bw和,delay和,loss和
                            # 添加到数组中
                            episode_bw = np.append(episode_bw, bw)
                            episode_delay = np.append(episode_delay, delay)
                            episode_loss = np.append(episode_loss, loss)
                            episode_length = np.append(episode_length, length)

                        # reward = self.env.calculate_path_score()
                        episode_reward = np.append(episode_reward, reward_temp)
                        episode_final_reward = np.append(episode_final_reward, reward)
                        episode_steps = np.append(episode_steps, self.env.step_num - 1)

                        self.print_train_info(episode, index, reward, link)
                        break

                    # 6. 状态改变
                    state = new_state

            self.writer.add_scalar('Episode/reward', episode_reward.mean(), episode)
            self.writer.add_scalar('Episode/final_reward', episode_final_reward.mean(), episode)
            self.writer.add_scalar('Episode/steps', episode_steps.mean(), episode)
            self.writer.add_scalar('Episode/bw', episode_bw.mean(), episode)
            self.writer.add_scalar('Episode/delay', episode_delay.mean(), episode)
            self.writer.add_scalar('Episode/loss', episode_loss.mean(), episode)
            self.writer.add_scalar('Episode/length', episode_length.mean(), episode)
            self.writer.add_scalar("learning_rate", self.rl.optimizer.param_groups[0]['lr'], episode)

            # self.rl.scheduler.step()

            all_episode_bw = np.append(all_episode_bw, episode_bw.mean())
            all_episode_delay = np.append(all_episode_delay, episode_delay.mean())
            all_episode_loss = np.append(all_episode_loss, episode_loss.mean())
            all_episode_length = np.append(all_episode_length, episode_length.mean())
            all_episode_reward = np.append(all_episode_reward, episode_reward.mean())
            all_episode_final_reward = np.append(all_episode_final_reward, episode_final_reward.mean())
            all_episode_steps = np.append(all_episode_steps, episode_steps.mean())

            self.rl.save_weight(self.name)

        self.save_episode_data(all_episode_reward, all_episode_final_reward, all_episode_steps, all_episode_bw,
                               all_episode_delay,
                               all_episode_loss, all_episode_length)

        self.logger.info(f'train over, cost time {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}')

    def set_reward_list_idx(self, idx):
        self.reward_list_idx = idx

    def save_episode_data(self, episode_reward, episode_final_reward, episode_steps, episode_bw, episode_delay,
                          episode_loss, episode_length):
        """
            保存数据
        :param episode_reward:episode回报值
        :param episode_final_reward:最终的回报值
        :param episode_steps:决策步数
        :param episode_bw:带宽
        :param episode_delay:时延
        :param episode_loss:丢包率
        :param episode_length:树长
        :return: None
        """
        np.save(self.data_path + "/episode_reward", episode_reward)
        np.save(self.data_path + "/episode_final_reward", episode_final_reward)
        np.save(self.data_path + "/episode_steps", episode_steps)
        np.save(self.data_path + "/episode_bw", episode_bw)
        np.save(self.data_path + "/episode_delay", episode_delay)
        np.save(self.data_path + "/episode_loss", episode_loss)
        np.save(self.data_path + "/episode_length", episode_length)

    def plot_episode_data(self, data, x_label, y_label, name):
        """
            画一个数据的图
        """
        x = range(len(data))
        plt.plot(x, data)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.savefig(self.image_path / f'{name}.png')

    def plot_compare_episode_data(self, data_list, x_label, y_label, name):
        """
            一个图画多个数据
        """
        for data in data_list:
            plt.plot(range(len(data)), data)

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()

        plt.savefig(self.image_path / f'{name}.png')

    def compare_test(self,
                     weight_file, figure=True):
        self.rl.change_model_mode('eval')

        pkl_cut_num = self.config.PKL_CUT_NUM
        pkl_start = self.config.PKL_START
        pkl_step = self.config.PKL_STEP
        start_node = self.config.start_node
        end_nodes = self.config.end_nodes

        episode_bw, kmb_bw, spanning_bw = [], [], []
        episode_delay, kmb_delay, spanning_delay = [], [], []
        episode_loss, kmb_loss, spanning_loss = [], [], []
        episode_length, kmb_length, spanning_length = [], [], []
        episode_alley = []

        self.rl.load_weight(weight_file, None)
        for index, pkl_path in enumerate(
                self.pkl_file_path_yield(self.config.pkl_weight_path, s=pkl_start, n=pkl_cut_num, step=pkl_step)):

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
                        bw, delay, loss, length, alley = self.env.get_route_params(
                            mode='test')  # 获得路径的所以链路bw和,delay和,loss和
                        # 添加到数组中
                        episode_bw.append(bw)
                        episode_delay.append(delay)
                        episode_loss.append(loss)
                        episode_length.append(length)
                        episode_alley.append(alley)

                        # kmb 算法
                        bw, delay, loss, length = self.get_kmb_params(self.env.graph, start_node, end_nodes)
                        kmb_bw.append(bw)
                        kmb_delay.append(delay)
                        kmb_loss.append(loss)
                        kmb_length.append(length)
                    break

                # 6. 状态改变
                state = new_state

        self.logger.info(
            f"{np.array(episode_bw).mean():.2f} & {np.array(episode_delay).mean():.2f} & {np.array(episode_loss).mean():.2e} & {np.array(episode_length).mean()} & {np.array(episode_alley).mean()}")

        if figure:
            self.plot_compare_figure(episode_bw, kmb_bw, "traffic", "mean bw", "bw")
            self.plot_compare_figure(episode_delay, kmb_delay, "traffic", "mean delay", "delay")
            self.plot_compare_figure(episode_loss, kmb_loss, "traffic", "mean loss", "loss")
            self.plot_compare_figure(episode_length, kmb_length, "traffic", "mean length", "length")

        bw_average, bw_max, bw_min = self.calculate_compare_result(episode_bw, kmb_bw)
        delay_average, delay_max, delay_min = self.calculate_compare_result(episode_delay, kmb_delay)
        loss_average, loss_max, loss_min = self.calculate_compare_result(episode_loss, kmb_loss)
        length_average, length_max, length_min = self.calculate_compare_result(episode_length, kmb_length)

        self.logger.info(f"bw: [bw_, delay_, loss_, length_]:\n average: {bw_average}, max: {bw_max}, min: {bw_min}")
        self.logger.info(
            f"delay: [bw_, delay_, loss_, length_]:\n average: {delay_average}, max: {delay_max}, min: {delay_min}")
        self.logger.info(
            f"loss: [bw_, delay_, loss_, length_]:\n average: {loss_average}, max: {loss_max}, min: {loss_min}")
        self.logger.info(
            f"length: [bw_, delay_, loss_, length_]:\n average: {length_average}, max: {length_max}, min: {length_min}")

        # self.plot_compare_figure_subplots(episode_bw, episode_delay, episode_loss, kmb_bw, kmb_delay, kmb_loss)

    def calculate_compare_result(self, episode, kmb):
        kmb_bw, kmb_delay, kmb_loss, kmb_length = [], [], [], []
        _average, _max, _min = [], [], []

        for data, episode_data in zip(kmb, episode):
            data = np.array(data)
            episode_data = np.array(episode_data)
            # mask = np.where(data != 0)

            # _delta = (episode_data - data[mask]) / data[mask]

            # kmb_bw.append(_delta[0])
            # kmb_delay.append(_delta[1])
            # kmb_loss.append(_delta[2])
            # kmb_length.append(_delta[3])
            if data[0] != 0 or episode_data == data[0]:
                kmb_bw.append((episode_data - data[0]) / (data[0] + 1e-30))
            if data[1] != 0 or episode_data == data[1]:
                kmb_delay.append((episode_data - data[1]) / (data[1] + 1e-30))
            if data[2] != 0 or episode_data == data[2]:
                kmb_loss.append((episode_data - data[2]) / (data[2] + 1e-30))
            if data[3] != 0 or episode_data == data[3]:
                kmb_length.append((episode_data - data[3]) / (data[3] + 1e-30))

        def _cal_metric(metric_list):
            _average.append(round(np.array(metric_list).mean() * 100, 2))
            _max.append(round(np.array(metric_list).max() * 100, 2))
            _min.append(round(np.array(metric_list).min() * 100, 2))

        _cal_metric(kmb_bw)
        _cal_metric(kmb_delay)
        _cal_metric(kmb_loss)
        _cal_metric(kmb_length)

        return _average, _max, _min

    def plot_compare_figure(self, rl_result, kmb_result, x_label, y_label, title):
        width = 0.18
        mode = args.compare_plot_mode
        kmb_bw = [kmb_result[i][0] for i in range(len(kmb_result))]
        kmb_delay = [kmb_result[i][1] for i in range(len(kmb_result))]
        kmb_loss = [kmb_result[i][2] for i in range(len(kmb_result))]
        kmb_length = [kmb_result[i][3] for i in range(len(kmb_result))]

        if mode == 'bar':
            plt.bar(range(len(kmb_result)), rl_result, width, label='rl')
            plt.bar([x + width for x in range(len(kmb_result))], kmb_bw, width, label='KMB_bw')
            plt.bar([x + 2 * width for x in range(len(kmb_result))], kmb_delay, width, label='KMB_delay')
            plt.bar([x + 3 * width for x in range(len(kmb_result))], kmb_loss, width, label='KMB_loss')
        else:
            plt.plot(range(len(kmb_result)), kmb_bw, ".-", label='KMB_bw', alpha=0.8)
            plt.plot(range(len(kmb_result)), kmb_delay, ".-", label='KMB_delay', alpha=0.8)
            plt.plot(range(len(kmb_result)), kmb_loss, ".-", label='KMB_loss', alpha=0.8)
            plt.plot(range(len(kmb_result)), rl_result, '*-', label='DRL-M4MR', alpha=0.8)

        # plt.xticks(range(len(kmb_result)), range(1, len(kmb_result) + 1))
        # plt.xticks(range(len(kmb_result)), range(len(kmb_result)), rotation=0, fontsize='small')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        # plt.title(title)
        plt.legend(bbox_to_anchor=(0., 1.0), loc='lower left', ncol=4, )

        _path = Path('./images')
        if _path.exists():
            plt.savefig(_path / f'{title}.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
        else:
            _path.mkdir(exist_ok=True)
            plt.savefig(_path / f'{title}.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
        if args.show:
            plt.show()
        plt.close()

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
            plt.savefig(_path / f'result.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
        else:
            _path.mkdir(exist_ok=True)
            plt.savefig(_path / f'result.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
        if args.show:
            plt.show()

    @staticmethod
    def read_data_path(data_path="./data"):
        """
            dict = {param_name: npy_path}
        """
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

    @staticmethod
    def get_diff_data_from_multi_file(data_path_list, param_name):
        """
            dict = {metric_name: npy_data}
        """
        data_dict = {"bw": [], "delay": [], "loss": [], 'length': [], "final_reward": [], "episode_reward": [],
                     "steps": [],
                     "legend": []}
        for path in data_path_list:
            data_dict['legend'].append(f"{param_name} " + path.name.split('_')[-1])
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
        """
            dict = {param_name: metric_name: npy_data}
        """
        data_path_dict = self.read_data_path(data_path)
        data_npy_dict = {}
        for k in data_path_dict.keys():
            data_npy_dict[k] = self.get_diff_data_from_multi_file(data_path_dict[k], param_name=k)

        return data_npy_dict

    def smooth(self, data_array, weight=0.8):
        # 一个类似 tensorboard smooth 功能的平滑滤波
        # https://dingguanglei.com/tensorboard-xia-smoothgong-neng-tan-jiu/
        last = data_array[0]
        smoothed = []
        for new in data_array:
            smoothed_val = last * weight + (1 - weight) * new
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    def plot_line_chart(self, data_list, para_name, title, x_label, y_label, legend_list):
        if data_list == []: return
        y_label_dict = {"bw": 'mean bottle bw', "delay": 'mean delay', "loss": 'mean loss', 'length': 'length',
                        "final_reward": "final reward", "episode_reward": "episode reward", "steps": 'steps'}
        # color = ["#F9D923", "#EB5353", "#36AE7C", "#187498"]
        color = ["#0848ae", "#17a858", "#df208c", "#e8710a"]

        i = 0
        fig, ax = plt.subplots(1, 1)
        # if y_label in ['final_reward', 'episode_reward', 'steps']:
        #     axins = ax.inset_axes((0.4, 0.4, 0.5, 0.4))
        # else:
        axins = None
        ylim0 = float('inf')
        ylim1 = 0
        for data, legend in zip(data_list, legend_list):
            x = range(len(data))
            xlim1 = len(data)
            ylim0 = np.min(data) if np.min(data) < ylim0 else ylim0
            ylim1 = np.max(data) if np.max(data) > ylim1 else ylim1
            ax.plot(x, data, alpha=0.1, color=color[i % 4])
            ax.plot(x, self.smooth(data), color=color[i % 4], label=legend)

            if axins is not None:
                axins.plot(x[-200:], data[-200:], alpha=0.1, color=color[i % 4])
                axins.plot(x[-200:], self.smooth(data)[-200:], color=color[i % 4])
            i += 1

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label_dict[y_label])
        ax.set_xlim(0, xlim1)
        # ax.set_ylim(ylim0, ylim1)
        # ax.legend(bbox_to_anchor=(0., 1.0), loc='lower left', ncol=len(legend_list))
        ax.legend()

        if axins is not None: mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)
        _p = Path(self.image_path / para_name)
        _p.mkdir(exist_ok=True)
        plt.savefig(_p / f'{title}.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
        if args.show:
            plt.show()
        plt.close(fig)

    def plot_all_line_chart_from_diff_param(self, data_path="./data"):
        data_npy_dict = self.get_compare_data(data_path)
        for param_name, metric_data in data_npy_dict.items():
            legend = metric_data.pop("legend")
            for metric_name, data in metric_data.items():
                self.plot_line_chart(data, param_name, metric_name, 'episodes', metric_name, legend)


def train_with_param(param_name, param_list):
    """
        对参数列表进行训练
    :param param_name:参数名
    :param param_list:参数的列表
    :return: None
    """
    param_dict = {'lr': Config.set_lr,
                  'nsteps': Config.set_nsteps,
                  'batchsize': Config.set_batchsize,
                  'egreedy': Config.set_egreedy,
                  'gamma': Config.set_gamma,
                  'updatefrequency': Config.set_update_frequency,
                  'rewardslist': Config.set_rewards,
                  'tau': Config.set_tau,
                  None: None}

    param_list_dict = {
        'lr': lr_list,
        'nsteps': n_step_list,
        'batchsize': batch_size_list,
        'egreedy': e_greedy_list,
        'gamma': gamma_list,
        'updatefrequency': update_steps_list,
        'rewardslist': rewardslist_step,
        'tau': tau_list,
    }

    assert param_name in param_dict.keys()

    if param_name is None:
        _name = f"[{Config.TIME}]_{Config.REWARD_DEFAULT}"
        train = Train(Config, MulticastEnv, DQN, MyMulticastNet3, name=_name)
        train.update()
    else:

        if param_list is None:
            param_list = param_list_dict[param_name]

        print(f"=================={param_name}==================")
        print(f"=================={param_list}==================")
        for param in param_list:
            set_param = param_dict[param_name]
            set_param(param)
            _name = f"[{Config.TIME}]_{param_name}_{param}"
            train = Train(Config, MulticastEnv, DQN, MyMulticastNet3, name=_name)
            train.update()


def test(pt_file_path):
    exp_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    _name = f"[{exp_time}]_test"
    train = Train(Config, MulticastEnv, DQN, MyMulticastNet3, name=_name, mode='eval')
    train.compare_test(pt_file_path)
    train.plot_all_line_chart_from_diff_param(data_path="./data")


def test_chart(pt_file_path):
    exp_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    _name = f"[{exp_time}]_test"
    train = Train(Config, MulticastEnv, DQN, MyMulticastNet3, name=_name, mode='eval')
    train.compare_test(pt_file_path, figure=False)


def test_diff_param():
    exp_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    _name = f"[{exp_time}]_test"
    train = Train(Config, MulticastEnv, DQN, MyMulticastNet3, name=_name, mode='eval')
    train.plot_all_line_chart_from_diff_param(data_path="./data")


if __name__ == '__main__':
    lr_list = [1e-4, 1e-5, 1e-6, 1e-7]
    n_step_list = [1, 2, 3, 4]
    batch_size_list = [8, 16, 32, 64]
    gamma_list = [0.3, 0.5, 0.7, 0.9]
    e_greedy_list = [100, 500, 1000]
    update_steps_list = [1, 10, 100, 1000]
    rewardslist_step = [[1.0, 0.01, -0.1, -1], [1.0, 0.1, -0.1, -1], [1.0, 1.0, -0.1, -1]]
    # rewardslist_step = [[1.0, 0.1, -0.1, -1], [1.0, 0.1, -0.01, -1], [1.0, 0.1, -0.001, -1]]
    tau_list = [1., 0.1, 0.01, 0.001]

    parser = argparse.ArgumentParser(description="train with different parameters")
    parser.add_argument('--param', '-p', default=None,
                        help="which param to train",
                        choices=['lr', 'nsteps', 'batchsize', 'egreedy', 'gamma', 'updatefrequency', 'rewardslist',
                                 'tau'])
    parser.add_argument("--param_list", '-pl', default=None, help="[num, num, num, ...]")
    parser.add_argument("--mode", '-m', default='test', choices=['train', 'test', 'testp', 'testc'],
                        help="train or test")
    parser.add_argument("--compare_plot_mode", default='line', choices=['bar', 'line'], help="compare plot")
    parser.add_argument("--pt", default="./saved_agents/policy_net-[20220629100456]_[1.0, 0.1, -0.1, -1].pt")
    parser.add_argument("--show", default=False)
    args = parser.parse_args()
    if args.mode == 'train':
        train_with_param(args.param, args.param_list)
    elif args.mode == 'test':
        test(args.pt)
    elif args.mode == 'testp':
        test_diff_param()
    elif args.mode == 'testc':
        test_chart(args.pt)

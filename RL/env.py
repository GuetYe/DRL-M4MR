# -*- coding: utf-8 -*-
# @File    : env.py
# @Date    : 2022-05-18
# @Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来-
# @From    :
import copy
import math
from itertools import zip_longest
from math import exp
from itertools import tee
from functools import reduce

import numpy as np
import networkx as nx

# from env_config import *
from config import Config


class MulticastEnv:
    """
        类初始化时，需要一个nx.Graph的图
        修改图时，调用self.modify_graph(graph)，修改self.graph属性
        调用环境前，使用self.reset(start, ends)，重置环境的属性、列表、矩阵等
        使用 self.step(link) 前进一步， 修改当前树，修改枝，修改路由矩阵、列表

        ---
        env = MulticastEnv(graph)
        env.reset(start_node, end_nodes)
        env.step(link)
    """

    def __init__(self, graph, numpy_type=Config.NUMPY_TYPE, normalize=True):
        """
        :param graph: nx.graph
        """
        self.graph = graph
        self.nodes = sorted(graph.nodes) if graph is not None else None
        self.edges = sorted(graph.edges, key=lambda x: (x[0], x[1])) if graph is not None else None
        self.numpy_type = numpy_type

        self.tree_nodes = set()  # 树节点
        self.route_graph = nx.Graph()  # 路由列表
        self.branches = set()  # 枝列表
        self.branches_graph = nx.Graph()
        self.mask = []

        self.n_actions = None

        self.start = None
        self.ends = None
        self.ends_constant = None

        self.adj_matrix = None
        self.route_matrix = None  # 路由矩阵
        self.state_matrix = None
        self.bw_matrix = None
        self.delay_matrix = None
        self.loss_matrix = None

        self.normal_bw_matrix = None
        self.normal_delay_matrix = None
        self.normal_loss_matrix = None

        self.step_normal_bw_matrix = None
        self.step_normal_delay_matrix = None
        self.step_normal_loss_matrix = None

        self.step_num = 1
        self.alley_num = 0

        self.max_bw = np.finfo(np.float32).eps
        self.min_bw = 0

        self.max_delay = np.finfo(np.float32).eps
        self.min_delay = 0

        self.max_loss = np.finfo(np.float32).eps
        self.min_loss = 0

        self.all_link_delay_sum = 0
        self.all_link_non_loss_prob = 0

        if graph is not None:
            # self.parse_graph_edge_data()  # 解析图
            self.set_nodes()
            self.set_edges()
            self.set_adj_matrix()  # 设置邻接矩阵

        self.done_reward = None
        self.step_reward = None
        self.hell_reward = None
        self.alley_reward = None

        self.discount = Config.DISCOUNT

        self.beta1 = Config.BETA1
        self.beta2 = Config.BETA2
        self.beta3 = Config.BETA3

        self.scale = normalize
        self.a_step = Config.A_STEP
        self.b_step = Config.B_STEP

        self.a_path = Config.A_PATH
        self.b_path = Config.B_PATH

        self.start_symbol = Config.START_SYMBOL
        self.end_symbol = Config.END_SYMBOL
        self.step_symbol = Config.STEP_SYMBOL
        self.branch_symbol = Config.BRANCH_SYMBOL

        self.set_reward_conf(Config.REWARD_DEFAULT)
        self.set_a_b(0, self.step_reward, 0, self.done_reward)

    def initialize_params(self):
        """
            重置 为空
        :return: None
        """
        self.tree_nodes = set()  # 树节点
        self.route_graph = nx.Graph()  # 树
        self.branches = set()  # 枝列表
        self.branches_graph = nx.Graph()
        self.mask = []

        self.n_actions = None

        self.start = None
        self.ends = None
        self.ends_constant = None

        self.adj_matrix = None
        self.route_matrix = None  # 路由矩阵
        self.state_matrix = None
        self.bw_matrix = None
        self.delay_matrix = None
        self.loss_matrix = None

        self.normal_bw_matrix = None
        self.normal_delay_matrix = None
        self.normal_loss_matrix = None

        self.step_normal_bw_matrix = None
        self.step_normal_delay_matrix = None
        self.step_normal_loss_matrix = None

        self.step_num = 1
        self.alley_num = 0

        self.max_bw = np.finfo(np.float32).eps
        self.min_bw = 0

        self.max_delay = np.finfo(np.float32).eps
        self.min_delay = 0

        self.max_loss = np.finfo(np.float32).eps
        self.min_loss = 0

        self.all_link_delay_sum = 0
        self.all_link_non_loss_prob = 0

    def set_params(self):
        """
            设置值
        :return: None
        """
        self.set_nodes()
        self.set_edges()
        self.set_adj_matrix()
        self.parse_graph_edge_data()

    def parse_graph_edge_data(self):
        """
            解析边的 bw， delay， loss 存入矩阵
        :return: bw_matrix, delay_matrix, loss_matrix
        """
        _num = len(self.nodes)
        # 全 - 1 矩阵
        bw_matrix = - np.ones((_num, _num), dtype=self.numpy_type)
        delay_matrix = - np.ones((_num, _num), dtype=self.numpy_type)
        loss_matrix = - np.ones((_num, _num), dtype=self.numpy_type)

        all_link_delay_sum, all_link_non_loss_prob = 0, 1
        for edge in self.graph.edges.data():
            _start, _end, _data = edge
            bw_matrix[_start - 1][_end - 1] = _data['bw']
            bw_matrix[_end - 1][_start - 1] = _data['bw']

            delay_matrix[_start - 1][_end - 1] = _data['delay']
            delay_matrix[_end - 1][_start - 1] = _data['delay']
            all_link_delay_sum += _data['delay']

            loss_matrix[_start - 1][_end - 1] = _data['loss']
            loss_matrix[_end - 1][_start - 1] = _data['loss']
            all_link_non_loss_prob *= 1 - _data['loss']

        maximum_spt_delay = list(nx.maximum_spanning_edges(self.graph, weight='delay'))
        self.all_link_delay_sum = reduce(lambda x, y: x + y, [link[2]['delay'] for link in maximum_spt_delay])
        maximum_spt_loss = list(nx.maximum_spanning_edges(self.graph, weight='loss'))
        self.all_link_non_loss_prob = reduce(lambda x, y: x * y, [1 - link[2]['loss'] for link in maximum_spt_loss])

        self.max_bw, self.min_bw, bw_matrix = self.get_non_minus_one_max_min(bw_matrix)
        self.max_delay, self.min_delay, delay_matrix = self.get_non_minus_one_max_min(delay_matrix)
        self.max_loss, self.min_loss, loss_matrix = self.get_non_minus_one_max_min(loss_matrix)

        self.bw_matrix = bw_matrix
        self.delay_matrix = delay_matrix
        self.loss_matrix = loss_matrix

        self.normal_bw_matrix, self.step_normal_bw_matrix = self.normalize_param_matrix(bw_matrix, self.max_bw,
                                                                                        self.min_bw)
        self.normal_delay_matrix, self.step_normal_delay_matrix = self.normalize_param_matrix(delay_matrix,
                                                                                              self.max_delay,
                                                                                              self.min_delay)
        self.normal_loss_matrix, self.step_normal_loss_matrix = self.normalize_param_matrix(loss_matrix, self.max_loss,
                                                                                            self.min_loss)
        return bw_matrix, delay_matrix, loss_matrix, all_link_delay_sum

    def normalize_param_matrix(self, matrix, max_value, min_value):
        """
            将矩阵按最大最小值正则 值在0到1之间
        """
        normal_m = (matrix - min_value) / (max_value + 1e-6)
        normal_m -= normal_m * np.identity(len(self.nodes))

        step_normal_m = self.a_step + normal_m * (self.b_step - self.a_step)
        return normal_m, step_normal_m

    def get_adjacent_link_params_max_min(self, node):
        """
            获得当前节点的周围链路的 参数 的最大最小值
        :param node: 当前节点
        :return: [(max_bw, min_bw), (max_delay, min_delay), (max_loss, min_loss)]
        """
        adj_index = self.get_node_adj_index_list(node)
        max_bw = self.bw_matrix[self.node_to_index(node), adj_index].max()
        min_bw = self.bw_matrix[self.node_to_index(node), adj_index].min()

        max_delay = self.delay_matrix[self.node_to_index(node), adj_index].max()
        min_delay = self.delay_matrix[self.node_to_index(node), adj_index].min()

        max_loss = self.loss_matrix[self.node_to_index(node), adj_index].max()
        min_loss = self.loss_matrix[self.node_to_index(node), adj_index].min()

        return [(min_bw, max_bw), (min_delay, max_delay), (min_loss, max_loss)]

    @staticmethod
    def get_non_minus_one_max_min(matrix):
        """
            返回除去-1的最大最小值
        :param matrix: 要求最大最小的矩阵
        :return: max, _min, matrix 【最大最小值， matrix将-1改为0】
        """
        non_minus_one_mask = np.where(matrix != -1)
        _max = matrix[non_minus_one_mask].max()
        _min = matrix[non_minus_one_mask].min()
        matrix[np.where(matrix == -1)] = 0
        return _max, _min, matrix

    def modify_graph(self, graph: nx.Graph):
        """
            修改 属性 并返回 graph
        :param graph: networkx的图
        :return: self.graph
        """
        self.graph = graph
        return self.graph

    def set_nodes(self):
        """
            修改并返回 nodes
        :return: self.nodes
        """

        self.nodes = sorted(self.graph.nodes)
        return self.nodes

    def set_edges(self):
        """
            设置 边
        :return: None
        """
        # [(1, 3), (1, 4), (1, 5), (1, 9), (1, 11),
        # (2, 4), (2, 5), (2, 6), (4, 5), (4, 6), (4, 7),
        # (4, 9), (4, 11), (5, 6), (5, 7), (5, 8), (7, 8),
        # (7, 9), (8, 14), (9, 10), (9, 13), (12, 13), (12, 14)]
        edges = [(e[0], e[1]) if e[0] < e[1] else (e[1], e[0]) for e in self.graph.edges]
        self.edges = sorted(edges, key=lambda x: (x[0], x[1]))

    def set_adj_matrix(self):
        """
            设置并返回邻接矩阵
        :return: adj_m
        """
        adj_m = nx.adjacency_matrix(self.graph, self.nodes).todense()
        self.adj_matrix = np.array(adj_m, dtype=self.numpy_type)
        return self.adj_matrix

    def read_pickle_and_modify(self, path):
        """
            读取图graph的pickle文件
            初始化所有参数
        :param path: pickle文件路径
        :return: 图 nx.graph
        """
        pkl_graph = nx.read_gpickle(path)
        self.modify_graph(pkl_graph)

        self.initialize_params()
        self.set_params()

        return pkl_graph

    @staticmethod
    def node_to_index(node):
        """
            节点从1起， 索引从0起，将节点号转为索引号
        :param node: 节点号
        :return: 索引号
        """
        return node - 1

    @staticmethod
    def index_to_node(index):
        """
            索引号转为节点号
        :param index: 索引号
        :return: 节点号
        """
        return index + 1

    def add_tree_node(self, node):
        """
            向树中添加节点
        """
        self.tree_nodes.add(node)

    def get_node_adj_index_list(self, node):
        """
            根据当前节点获得邻居节点的索引号
        :param node: 节点序号
        :return: 邻居节点的索引号列表
        """
        index = self.node_to_index(node)
        adj_ids = np.nonzero(self.adj_matrix[index])  # return tuple
        adj_index_list = adj_ids[0].tolist()
        return adj_index_list

    def modify_branches(self, node):
        """
            修改枝列表
            修改之前需要先 将满足条件的node添加到tree node, self.add_tree_node(node)
            prim算法中有可添加链路方法
            2022/3/17 大改
        :return: None
        """
        # A. 修改 branch
        # 取单行, 即为node的邻居节点行, 是邻居节点则为1, 否则是0
        adj_ids_list = self.get_node_adj_index_list(node)
        adj_nodes = set(map(self.index_to_node, adj_ids_list))  # 序号转为node
        # 1. 更新branches中的节点
        self.branches.update(adj_nodes)
        # 2. 移除branches中的tree集合的节点
        self.branches.difference_update(self.tree_nodes)

        # B. 修改 branches_graph
        # 1. 枝节点的周围节点添加进去
        for u, v in zip_longest([node], list(adj_nodes), fillvalue=node):  # zip_longest
            if v in self.tree_nodes:
                self.branches_graph.remove_edge(u, v)
            else:
                self.branches_graph.add_edge(u, v)

    def get_branches_matrix(self):
        """
            获得当前枝的邻接矩阵表示
        """
        branches_m = np.zeros((len(self.nodes), len(self.nodes)), dtype=self.numpy_type)
        for e in self.branches_graph.edges:
            u, v = e
            u = self.node_to_index(u)
            v = self.node_to_index(v)
            branches_m[u][v] = self.branch_symbol
            branches_m[v][u] = self.branch_symbol
        return branches_m

    def add_to_route_graph(self, link):
        """
            添加link到 self.route_graph中
        :param link: 链路
        :return: None
        """
        self.route_graph.add_edge(*link)

    def get_mask_of_current_tree_branch(self):
        """
            获得是当前树的branch的mask
            (1, 0, 0, 1, ...)
        :return: mask
        """
        _branches = self.branches_graph.edges
        edges = self.edges  # 这个list要保证不变
        mask = np.zeros(len(self.graph.edges), dtype=bool)
        for edge in _branches:
            if edge in edges:
                # 将索引位置设置为True
                mask[edges.index(edge)] = True
            elif edge[::-1] in edges:
                mask[edges.index(edge[::-1])] = True
            else:
                raise IndexError("edge not in branches_graph")

        self.mask = mask

        return mask

    def _judge_link(self, link):
        """
            弃用，link[1]错误问题
            判断link是否满足当前的情况，是否是当前树的枝
        :param link: 下一步链路
        :return: True or False
        """
        # 是否是树枝，否则返回None
        end_node = link[1]
        if end_node in self.branches:
            return True
        else:
            return False

    def judge_end_node(self, link):
        """
            判断 link 中哪个是 长出来的新枝，并修正link方向
        :param link: 下一跳的link
        :return: tree_node, branch_node
        """
        tree_node = self.tree_nodes & set(link)  # 取交
        if len(tree_node) == 1:
            branch_node = set(link) - self.tree_nodes  # link中包含而tree_nodes中不包含

            try:
                return list(tree_node)[0], list(branch_node)[0]
            except IndexError:
                raise IndexError(f"({link}, {tree_node}, {branch_node}, {self.tree_nodes})")
        else:
            return None

    def judge_link(self, link):
        """
            判断 link 是否是枝
        :param link: 下一跳的link
        :return: True or False
        """
        if link in self.branches_graph.edges:
            return True
        else:
            return False

    def reset(self, start, ends):
        """
            构建初始的路径矩阵，
        :param start: 1, 2,... 比标号多1， 索引标号从0 起
        :param ends: list
        :return: route_matrix: numpy matrix
        """
        if self.graph is None:
            raise Exception("graph is None")

        self.start = copy.deepcopy(start)
        self.ends = copy.deepcopy(ends)
        self.ends_constant = copy.deepcopy(ends)

        _num = len(self.nodes)
        # 全 0 矩阵
        route_matrix = np.zeros((_num, _num), dtype=self.numpy_type)
        # 将路径矩阵的源节点设置为 1
        _idx = self.node_to_index(start)
        route_matrix[_idx, _idx] = self.start_symbol
        # 将目的节点设置为 -1
        for end in self.ends:
            _idx = self.node_to_index(end)
            route_matrix[_idx, _idx] = self.end_symbol

        self.route_matrix = route_matrix
        self.state_matrix = route_matrix

        # 添加源节点到树
        self.add_tree_node(self.start)
        # 添加源节点的枝
        self.modify_branches(self.start)
        # 获得当前树的枝mask
        self.get_mask_of_current_tree_branch()

        return route_matrix

    def step(self, link):
        """
            更新，将新的路径放入route_matrix
            2022/3/17 不是tree中的节点作为 end_node 而不是 link[1]作为end_node
        :param link: 链路 如(1, 2)
        :return: state_: numpy.matrix, reward_score: float, route_done: bool
        """
        assert self.hell_reward is not None
        # 判断link是否满足当前的情况，是否是当前 树的枝
        link = self.judge_end_node(link)
        if link:
            tree_node, branch_node = link
            # 1. 路由矩阵打上记号
            self.route_matrix[self.node_to_index(tree_node)][self.node_to_index(branch_node)] = self.step_symbol
            self.route_matrix[self.node_to_index(branch_node)][self.node_to_index(tree_node)] = self.step_symbol
            # 2.1 添加目的节点到树节点集合中
            self.add_tree_node(branch_node)
            # 2.2 修改枝列表
            self.modify_branches(branch_node)
            # 2.3 获得当前树的枝mask
            self.get_mask_of_current_tree_branch()
            # 3. 链路添加到路由列表中
            self.add_to_route_graph((tree_node, branch_node))

            # 判断是结束了还是往前进了一步
            _judge_flag = self._judge_end(branch_node)
            if _judge_flag == 'ALL':
                route_done = True
                state_ = None  # 这个动作后状态，无动作
                reward_score = self.calculate_path_score()
                # reward_score = self.calculate_link_reward(link)
                # reward_score = (self.discount ** self.step_num) * reward_score
                # reward_score = self.done_reward + self.step_num * self.step_reward
                alley, _ = self.find_blind_alley()
                # reward_score += alley * self.alley_reward
                self.alley_num = alley

            elif _judge_flag == 'PART':
                route_done = False
                state_ = self.route_matrix
                reward_score = self.calculate_link_reward(link)
                # reward_score = self.calculate_path_score()
                # alley = self.find_blind_alley()
                # reward_score += alley * self.alley_reward
            elif _judge_flag == "NOT":
                route_done = False
                state_ = self.route_matrix
                reward_score = self.calculate_link_reward(link)
            else:
                raise Exception("link judge error")
        else:
            _judge_flag = "HELL"
            route_done = False
            state_ = self.route_matrix
            reward_score = self.hell_reward

        self.step_num += 1
        return state_, reward_score, route_done, _judge_flag

    def _judge_end(self, next_node):
        """
            判断是否是已经结束。
            如果目的节点空了，表示已经找到所有目的节点
        :param next_node: 下一跳
        :return: "ALL", "PART", "NOT"
        """
        assert next_node is not None, "next_node is None"
        if next_node in self.ends:
            _i = self.ends.index(next_node)
            self.ends.pop(_i)

            # 目的节点列表为空
            if len(self.ends) == 0:
                return "ALL"
            else:
                return "PART"
        return "NOT"

    def reward_score(self, link):
        """
            奖励
        :param link: 链路
        :return: reward
        """
        score = self.calculate_link_reward(link)
        return score

    def step_reward_exp(self):
        """
            step_reward * e**(1/nodes_num * step_num)
        :return: score
        """
        score = self.step_reward * exp(1 / len(self.nodes) * self.step_num - 1)
        return score

    def discount_reward(self, score):
        """
            对reward进行discount 处理 score = score * discount ** step_num
        :param score: reward
        :return: score
        """
        score *= self.discount ** (self.step_num - 1)
        return score

    def link_reward_func(self, bw, delay, loss, b):
        """
            计算reward
            R = β1 * bw + β2 * (1 - delay) + β3 * (1 - loss)
            :param bw: 带宽
            :param delay: 时延
            :param loss: 丢包率
            :param b: 上限
        """
        return self.beta1 * bw + self.beta2 * (b - delay) + self.beta3 * (b - loss)
        # return -(self.beta1 * (b - bw) + self.beta2 * delay + self.beta3 * loss)

    def path_reward_func(self, bw, delay, loss, b):
        return self.beta1 * bw + self.beta2 * (b - delay) + self.beta3 * loss
        # return -(self.beta1 * (b - bw) + self.beta2 * delay + self.beta3 * (b - loss))

    def calculate_link_reward(self, link):
        """
            计算每步选择link reward R
        :return: reward
        """
        e0, e1 = self.node_to_index(link[0]), self.node_to_index(link[1])
        bw_hat = self.step_normal_bw_matrix[e0, e1]
        delay_hat = self.step_normal_delay_matrix[e0, e1]
        loss_hat = self.step_normal_loss_matrix[e0, e1]
        r = self.link_reward_func(bw_hat, delay_hat, loss_hat, self.b_step)
        return r

    def calculate_path_score(self):
        """
            计算路径的回报
        :return: reward
        """
        non_losses = np.array([])
        delays = np.array([])

        e2e_bw, e2e_delay, e2e_bw_hat, e2e_delay_hat = self.find_end_to_end_max_bw_delay(self.route_graph, self.start,
                                                                                         self.ends_constant)

        for link in self.route_graph.edges:
            delay = self.graph[link[0]][link[1]]['delay']
            loss = self.graph[link[0]][link[1]]['loss']
            delays = np.append(delays, delay)
            non_losses = np.append(non_losses, 1 - loss)
        # delay_hat = e2e_delay_hat
        num = len(self.ends_constant) + 1
        delay_hat = self.min_max_normalize(delays.sum(), self.min_delay * num, self.all_link_delay_sum, a=self.a_path,
                                           b=self.b_path)

        # prob sum
        _min = 1 - self.max_loss
        _max = 1 - self.min_loss
        non_loss_hat = non_losses.prod()
        # non_loss_hat = self.min_max_normalize(non_losses.prod(), _min ** num, _max, a=self.a_path, b=self.b_path)

        # if delays.sum() < self.min_delay * num:
        #     raise ValueError("delays.sum() < self.min_delay * num")

        r = self.path_reward_func(e2e_bw_hat, delay_hat, non_loss_hat, self.b_path)
        return r

    def find_end_to_end_max_bw(self, route_graph, start, ends):
        """
            找到端对端的最大剩余带宽
        :param route_graph:组播树
        :param start:源节点
        :param ends:目的节点
        :return bws:端对端的带宽列表
        """
        bws = np.array([])
        for end in ends:
            p = nx.shortest_path(route_graph, source=start, target=end)
            bw = self.max_bw

            def pairwise(iterable):
                a, b = tee(iterable)
                next(b, None)
                return zip(a, b)

            for e in pairwise(p):
                if self.graph.edges[e[0], e[1]]['bw'] < bw:
                    bw = self.graph.edges[e[0], e[1]]['bw']
                else:
                    pass
            bws = np.append(bws, bw)
        return bws

    def find_end_to_end_max_bw_delay(self, route_graph, start, ends):
        """
            从源节点到各个目的节点路径，分别的最大剩余带宽 时延
        :param route_graph:组播树
        :param start:源节点
        :param ends:目的节点
        :return:最大剩余带宽列表 时延列表 bws, delays, bws_hat, delays_hat
        """
        bws = np.array([])
        delays = np.array([])
        delays_hat = np.array([])
        for end in ends:
            if end in route_graph.nodes:
                p = nx.shortest_path(route_graph, source=start, target=end)
                bw = self.max_bw
                delay = 0

                def pairwise(iterable):
                    a, b = tee(iterable)
                    next(b, None)
                    return zip(a, b)

                for e in pairwise(p):
                    if self.graph.edges[e[0], e[1]]['bw'] < bw:
                        bw = self.graph.edges[e[0], e[1]]['bw']
                    else:
                        pass
                    delay += self.graph.edges[e[0], e[1]]['delay']

                bws = np.append(bws, bw)
                delays = np.append(delays, delay)

                num = len(self.edges)
                delay_hat = self.min_max_normalize(delay, self.min_delay * num, self.all_link_delay_sum, a=self.a_path,
                                                   b=self.b_path)

                delays_hat = np.append(delays_hat, delay_hat)
            else:
                continue
        bws_hat = self.min_max_normalize(bws.mean(), self.min_bw, self.max_bw, a=self.a_path, b=self.b_path)
        delays_hat = delays_hat.max()

        return bws, delays, bws_hat, delays_hat

    def min_max_normalize(self, x, min_value, max_value, a=None, b=None):
        """
            最大最小标准化
        :param x: 要标准化的值
        :param min_value: 最小值
        :param max_value: 最大值
        :param a: 下限
        :param b: 上限
        :return: 标准化后的x_hat
        """
        if a is None:
            a = self.a_step
        if b is None:
            b = self.b_step

        # 加一个很小的值
        x_normal = (x - min_value) / (max_value - min_value + np.finfo(np.float32).eps)
        x_hat = a + x_normal * (b - a)
        return x_hat.astype(self.numpy_type)

    def get_route_params(self, mode='train'):
        """
            计算路径的剩余带宽, 时延, 丢包率
        :return : 路径的剩余带宽, 时延, 丢包率
        """
        bw, delay, loss = 0, 0, 0
        num = 0
        alley, r_route_graph = self.find_blind_alley()
        for r in r_route_graph.edges:
            bw += self.graph[r[0]][r[1]]["bw"]
            delay += self.graph[r[0]][r[1]]["delay"]
            loss += self.graph[r[0]][r[1]]["loss"]
            num += 1

        bws = self.find_end_to_end_max_bw(self.route_graph, self.start, self.ends_constant)
        if mode == 'train':
            length = len(self.route_graph.edges)
        elif mode == 'test':
            length = len(r_route_graph.edges)

        return bws.mean(), delay / num, loss / num, length, alley

    def map_action(self, action):
        """
            根据动作的标号，返回选择的链路
            如 action=1, 返回(1, 2)

            2022/3/17 large modification

        :param action: 动作标号
        :return: 下一跳
        """
        edge = self.edges[action]
        return edge

    def find_blind_alley(self):
        """
            找到死角
        """
        alley = 0
        r_graph = self.route_graph.copy()
        while True:
            del_node = []
            for pair in r_graph.degree:
                node, degree = pair
                if degree == 1 and node not in self.ends_constant and node != self.start:
                    alley += 1
                    del_node.append(node)
            for node in del_node:
                r_graph.remove_node(node)

            if not del_node:
                break

        return alley, r_graph

    def set_reward_conf(self, rewards_list):
        """
            设置奖励值
        """
        done_reward, step_reward, alley_reward, hell_reward = rewards_list
        self.done_reward = done_reward
        self.step_reward = step_reward
        self.hell_reward = hell_reward
        self.alley_reward = alley_reward

    def set_a_b(self, a_step, b_step, a_path, b_path):
        """
            设置归一化ab的值
        """

        self.a_step = a_step
        self.a_path = a_path
        self.b_step = abs(b_step)
        self.b_path = abs(b_path)

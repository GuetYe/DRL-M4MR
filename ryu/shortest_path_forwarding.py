# -*- coding: utf-8 -*-
# @File    : shortest_path_forwarding.py
# @Date    : 2021-08-13
# @Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来-
import copy
import sys
import time
import csv
from pathlib import Path
import pickle
import json

import networkx as nx
from networkx.algorithms.approximation import steiner_tree

from ryu.base import app_manager
from ryu.base.app_manager import lookup_service_brick
from ryu.ofproto import ofproto_v1_3
from ryu.controller import ofp_event
from ryu.controller.handler import set_ev_cls, MAIN_DISPATCHER, DEAD_DISPATCHER, CONFIG_DISPATCHER
from ryu.lib import hub
from ryu.lib.packet import packet
from ryu.lib.packet import arp, ipv4, ethernet, igmp
from ryu.lib import igmplib, mac

import setting
import network_structure
import network_monitor
import network_delay
from plot_graphs import plot_x_y

from RL.train import Train
from RL.config import Config
from RL.rl import DQN
from RL.env import MulticastEnv
from RL.net import MyMulticastNet3


# ETHERNET = ethernet.ethernet.__name__
# ETHERNET_MULTICAST = "ff:ff:ff:ff:ff:ff"
# ARP = arp.arp.__name__


class ShortestPathForwarding(app_manager.RyuApp):
    OFP_VERSION = [ofproto_v1_3.OFP_VERSION]

    # _CONTEXTS = {
    #     'discovery': network_structure.NetworkStructure,
    #     'monitor': network_monitor.NetworkMonitor,
    #     'detector': network_delay.NetworkDelayDetector
    # }

    def __init__(self, *args, **kwargs):
        super(ShortestPathForwarding, self).__init__(*args, **kwargs)
        self.discovery = lookup_service_brick('discovery')
        self.monitor = lookup_service_brick('monitor')
        self.detector = lookup_service_brick('detector')
        self.name = 'shortest_path_forwarding'
        self.count = 0
        self.finish_flag = False
        self.start_flag = False
        self.now_pkl = None

        now_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.weight_dir = './weight/' + now_time
        Path.mkdir(Path(self.weight_dir), exist_ok=True, parents=True)
        self.pickle_dir = './pickle' + '/' + now_time
        Path(self.pickle_dir).mkdir(exist_ok=True, parents=True)

        self.drl_multicast = Train(Config, MulticastEnv, DQN, MyMulticastNet3,
                                   name=f'{time.strftime("%Y%m%d%H%M%S", time.localtime())}', mode='eval')

        self.drl_multicast.predict(r".\RL\saved_agents\init_pkl.pkl")
        # self.arp_table = {}
        # self.sw = {}
        # self.mac_to_port = {}
        # self.graph = self.discovery.graph
        # self.datapaths_table = self.monitor.datapaths_table  # {dpid: datapath}
        # self.access_table = self.discovery.access_table  # {(dpid, in_port): (src_ip, src_mac)}
        # self.link_port_table = self.discovery.link_port_table  # {(src.dpid, dst.dpid): (src.port_no, dst.port_no)}
        # self.switch_all_ports_table = self.discovery.switch_all_ports_table  # {dpid: {port_no, ...}}
        self.my_logger = self.logger.info if setting.LOGGER else print

        self.plot_info = {}

        self.shortest_thread = hub.spawn(self.super_schedule)
        self.create_graph_thread = hub.spawn(self.check_finish)

    def super_schedule(self):
        """
            总调用线程，
            self.discovery.scheduler() 网络探测
            self.monitor.scheduler()  网络带宽，loss监测
            self.detector.scheduler()  时延检测
            self.create_weight_graph()  刷新图权重
            self.save_links_weight(self.count)  保存图信息
        """
        while True:
            if self.finish_flag:
                break

            if self.start_flag:
                self.count += 1

            if not self.discovery.first_flag:
                print(f'[save links weight {self.count}]')
                self.save_links_weight(self.count)
            hub.sleep(setting.SCHEDULE_PERIOD)

    def check_finish(self):
        while True:
            if Path(setting.finish_time_file).exists():
                with open(setting.finish_time_file, 'r') as f:
                    _read = json.load(f)

                    try:
                        # print("shortest--->check_finish", self.start_flag, self.finish_flag)
                        self.start_flag = _read["start_save_flag"]
                        self.finish_flag = _read['finish_flag']
                    except KeyError:
                        pass
            else:
                pass
            hub.sleep(setting.SCHEDULE_PERIOD)

    def save_links_weight(self, count):
        """
            保存graph信息， csv和pkl
        """
        name = f"{count}-" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.save_csv_graph(name)
        self.save_pickle_graph(name)
        self.save_pkl_plot_info()

    def save_pickle_graph(self, name):
        """
            保存图信息的pkl文件
            ./pkl/now_time/name.pkl
        """
        _path = self.pickle_dir / Path(name + '.pkl')
        _graph = self.discovery.graph.copy()
        nx.write_gpickle(_graph, _path)

    def save_csv_graph(self, name):
        """
            保存图信息的csv文件
            ./weight/now_time/name.csv
        """
        with open(self.weight_dir + "/" + name + '.csv', 'w+', newline='') as f:
            f_csv = csv.writer(f)
            _graph = self.discovery.graph.copy()
            f_csv.writerows(list(_graph.edges(data=True)))

            # save plot info
            # self.add_plot_info((start_, end_), weight)

    def add_plot_info(self, edge, weight):
        """
            save plot info
        """
        bw, delay, loss = weight['bw'], weight['delay'], weight['loss']
        try:
            self.plot_info[edge]["bw"].append(bw)
            self.plot_info[edge]["delay"].append(delay)
            self.plot_info[edge]["loss"].append(loss)
        except Exception as e:
            print(e)
            self.plot_info.setdefault(edge, {"bw": [bw], "delay": [delay], "loss": [loss]})

    def save_pkl_plot_info(self):
        with open(self.pickle_dir + "/" + "plot_info" + '.pkl', 'wb+') as f:
            pickle.dump(self.plot_info, f, 0)

        self.now_pkl = self.pickle_dir + "/" + "plot_info" + '.pkl'

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        """
            处理PacketIn事件
            1. arp包 是否已经记录，如果没有记录则泛洪
            2. ipv4包 下发流表
        """
        # print("shortest---> _packet_in_handler: PacketIn")
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        # print("shortest---> _packet_in_handler: pkt:\n  ", pkt)

        arp_pkt = pkt.get_protocol(arp.arp)
        ipv4_pkt = pkt.get_protocol(ipv4.ipv4)

        if isinstance(ipv4_pkt, ipv4.ipv4):
            # print("shortest--->_packet_in_handler: ==IPv4 processing")
            if ipv4_pkt.src == '0.0.0.0' or ipv4_pkt.dst == '224.0.0.22':
                return None

            if len(pkt.get_protocols(ethernet.ethernet)):
                eth_type = pkt.get_protocols(ethernet.ethernet)[0].ethertype
                # 根据路径下发流表
                self.get_shortest_path(msg, eth_type, ipv4_pkt.src, ipv4_pkt.dst)

    def get_shortest_path(self, msg, eth_type, src_ip, dst_ip):
        """ 根据消息计算最短路径"""
        datapath = msg.datapath
        in_port = msg.match['in_port']

        # 1. 找出位置
        src_dst_switches = self.get_switches(datapath.id, in_port, src_ip, dst_ip)
        if src_dst_switches:
            src_switch, dst_switch = src_dst_switches
            if len(dst_switch) == 1:
                dst_switch = dst_switch[0]
                # 2.1 计算最短路径
                if src_switch == dst_switch:
                    return
                path = self.get_path(src_switch, dst_switch)
                self.logger.info("[PATH]%s<-->%s: %s" % (src_switch, dst_switch, path))
                # 3.1 下发流表
                self.install_flow(path, eth_type, src_ip, dst_ip, in_port, msg.buffer_id, msg.data)
            elif len(dst_switch) > 1:
                # 2.2 计算组播路径
                path = self.calculate_multicast_path(src_switch, dst_switch)
                self.logger.info("[MULTICAST_PATH]%s<-->%s: %s" % (src_switch, dst_switch, path))
                # 3.2 下发流表
                self.install_multicast_flow(path, src_switch, dst_switch, eth_type, src_ip, dst_ip, in_port,
                                            msg.buffer_id, msg.data)
        else:
            print("shortest--->get_shortest_path: src_dst_switches", src_dst_switches)

    def get_multicast_shortest_path(self, msg, eth_type, src_ip, dst_ip):
        """
            计算组播的最短路径
        """
        datapath = msg.datapath
        in_port = msg.match['in_port']

        # 1. 找出位置
        src_dst_switches = self.get_switches(datapath.id, in_port, src_ip, dst_ip)
        if src_dst_switches:
            src_switch, dst_switch = src_dst_switches
            # 2.2 计算组播路径
            path = self.calculate_multicast_path(src_switch, dst_switch)
            # 3.2 下发流表
            self.install_multicast_flow(path, src_switch, dst_switch, eth_type, src_ip, dst_ip, in_port,
                                        msg.buffer_id, msg.data)

        else:
            print("shortest--->get_shortest_path: src_dst_switches", src_dst_switches)

    # 构造输出的包
    def _build_packet_out(self, datapath, buffer_id, src_port, dst_port, data):
        """ 构造输出的包"""
        actions = []
        if dst_port:
            actions.append(datapath.ofproto_parser.OFPActionOutput(dst_port))

        msg_data = None
        if buffer_id == datapath.ofproto.OFP_NO_BUFFER:
            if data is None:
                return None
            msg_data = data

        out = datapath.ofproto_parser.OFPPacketOut(datapath=datapath, buffer_id=buffer_id,
                                                   data=msg_data, in_port=src_port, actions=actions)

        return out

    # 获得源交换机dpid和目标交换机的dpid
    def get_switches(self, dpid, in_port, src_ip, dst_ip):
        """ 根据src_ip求得dpid"""
        src_switch = dpid
        dst_switch = list()

        # 获得src switch的位置
        src_location = self.discovery.get_host_ip_location(src_ip)  # (dpid, in_port)
        if in_port in self.discovery.not_use_ports[dpid]:  # {dpid: {port_no, ...}}
            # print(f"shortest--->get_switches: src_location == {(dpid, in_port)}")
            if (dpid, in_port) == src_location:
                src_switch = src_location[0]
            else:
                return None

        # if dst ip is multicast address then
        # 如果是设定的组播地址，那么
        if dst_ip in setting.DST_MULTICAST_IP.keys():
            dst_group_ip = setting.DST_MULTICAST_IP[dst_ip]
            for other_ip in dst_group_ip:
                dst_location = self.discovery.get_host_ip_location(other_ip)
                if dst_location:
                    dst_switch.append(dst_location[0])
            print(f"shortest--->get_switches: dst_switch == {dst_switch}")
        else:
            dst_location = self.discovery.get_host_ip_location(dst_ip)
            if dst_location:
                dst_switch.append(dst_location[0])

        return src_switch, dst_switch

    def get_port(self, dst_ip):
        """ 根据目的ip获得出去的端口"""
        for key in self.discovery.access_table.keys():  # {(dpid, in_port): (src_ip, src_mac)}
            if dst_ip == self.discovery.access_table[key][0]:
                dst_port = key[1]
                return dst_port
        return None

    def get_port_by_dpid_ip(self, dpid, dst_ip):
        """
            通过 dpid 和 dst_ip 获得 端口
        """
        for key, value in self.discovery.access_table.keys():
            if dpid == key[0] and dst_ip == value[0]:
                port = key[1]
                return port
        return None

    def get_port_pair(self, src_dpid, dst_dpid):
        """ 根据源dpid和目的dpicalculate_all_nodes_shortest_pathsd获得src.port_no, dst.port_no"""
        if (src_dpid, dst_dpid) in self.discovery.link_port_table:
            return self.discovery.link_port_table[(src_dpid, dst_dpid)]
        else:
            print("shortest--->get_port_pair: dpid: %s -> dpid: %s is not in links", (src_dpid, dst_dpid))
            return None

    def get_path(self, src_dpid, dst_dpid):
        """ 计算最短路径"""

        shortest_path = self.discovery.shortest_path_table[(src_dpid, dst_dpid)]
        return shortest_path

    def calculate_multicast_path(self, src, dsts, algorithms="st"):
        """
            根据 algorithms 计算steiner树
            src: 源点 int
            dsts： 目的节点 list
            return nx.graph : st_path steiner tree
        """
        graph = self.discovery.graph.copy()
        terminals = [src] + dsts
        st_path = None
        if algorithms == 'st':
            # 应该是KMB算法
            st_path = steiner_tree(graph, terminals, weight=setting.WEIGHT)

        elif algorithms == 'drl':
            st_path = self.drl_multicast.predict(self.now_pkl)

        return st_path

    @staticmethod
    def dfs_path(tree, src):
        # ：：生成器只能遍历一次：：
        dfs_path = list(nx.dfs_edges(tree, src))
        print("shortest--->calculate_multicast_path, dfs_path:", dfs_path)
        return dfs_path

    @staticmethod
    def path_pair_to_one_path(path, src, dsts):
        """
            将路径对转换为 一个一个路径
            src:1 , dst: 2,7,8
            path: dfs
            [(1, 7), (7, 2), (1, 16), (16, 5), (5, 8)]
            then return
            [[1, 7], [1, 7, 2], [1, 16, 5, 8]]
        """
        temp_path = list()
        all_one_path = list()
        for pair in path:
            pre, nex = pair

            if pre == src:
                temp_path = [pre]
            elif pre in temp_path:
                temp_path = temp_path[: temp_path.index(pre) + 1]

            temp_path.append(nex)

            if nex in dsts:
                all_one_path.append(temp_path)

        return all_one_path

    @staticmethod
    def get_adjacency_switch(st_path):
        """
            通过graph的adjacency获得邻居节点
        :param st_path: nx graph
        :return: adj
        """
        adj = {}
        for n, nbrsdict in st_path.adjacency():
            adj.setdefault(n, list(nbrsdict.keys()))
        return adj

    def get_father_son_switch(self, st_path, source):
        """
            获得每个中间节点的父节点和子节点
        :param st_path: steiner tree
        :param source: 根节点
        :return: father_son_dict {curr: [father, [son]]}
        """
        # 邻居节点 {n : adj}
        adj = self.get_adjacency_switch(st_path)
        # 深度优先遍历
        # _path = nx.dfs_edges(st_path, source=source)
        _path = nx.bfs_edges(st_path, source=source)
        father_son_dict = {}
        father_son_dict.setdefault(source, [[], []])

        for nodes in _path:
            father, curr = nodes
            idx = adj[curr].index(father)
            adj[curr].pop(idx)
            son = adj[curr]
            father_son_dict.setdefault(curr, [father, son])
            if father == source:
                father_son_dict[source][1].append(curr)
        return father_son_dict

    def install_flow(self, path, eth_type, src_ip, dst_ip, in_port, buffer_id, data=None):
        """ 有多种情况需要考虑，即走哪个端口"""
        if path is None or len(path) == 0:
            print("shortest--->install_flow: Path Error")
            return
        else:
            first_dp = self.monitor.datapaths_table[path[0]]

            if len(path) > 2:
                # print("shortest--->install_flow: len(path) > 2")
                for i in range(1, len(path) - 1):
                    port_pair = self.get_port_pair(path[i - 1], path[i])
                    port_pair_next = self.get_port_pair(path[i], path[i + 1])
                    # print("shortest--->install_flow: len(path) > 2 port_pair, port_pair_next", port_pair,
                    #       port_pair_next)
                    if port_pair and port_pair_next:
                        src_port, dst_port = port_pair[1], port_pair_next[0]  # 同一个交换机的不同口, 见图
                        datapath = self.monitor.datapaths_table[path[i]]
                        # 下发正向流表
                        self.send_flow_mod(datapath, eth_type, src_ip, dst_ip, src_port, dst_port)
                        # 下发反向流表
                        self.send_flow_mod(datapath, eth_type, dst_ip, src_ip, dst_port, src_port)
                    else:
                        print(f"shortestERROR--->install_flow: len(path) > 2 "
                              f"path_0, path_1, port_pair: {path[i - 1], path[i], port_pair}, "
                              f"path_1, path_2, next_port_pair: {path[i], path[i + 1], port_pair_next}")
                        return

            if len(path) > 1:
                # print("shortest--->install_flow: len(path) == 2")
                port_pair = self.get_port_pair(path[-2], path[-1])

                if port_pair is None:
                    print("shortest--->install_flow: port not found")
                    return

                src_port = port_pair[1]
                dst_port = self.get_port(dst_ip)

                if dst_port is None:
                    print("shortest--->install_flow: Last port is not found")
                    return

                last_dp = self.monitor.datapaths_table[path[-1]]
                self.send_flow_mod(last_dp, eth_type, src_ip, dst_ip, src_port, dst_port)
                self.send_flow_mod(last_dp, eth_type, dst_ip, src_ip, dst_port, src_port)

                port_pair = self.get_port_pair(path[0], path[1])
                if port_pair is None:
                    print("shortest--->install_flow: port not found in -2 switch")
                    return

                out_port = port_pair[0]
                self.send_flow_mod(first_dp, eth_type, src_ip, dst_ip, in_port, out_port)
                self.send_flow_mod(first_dp, eth_type, dst_ip, src_ip, out_port, in_port)
                self.send_packet_out(first_dp, buffer_id, in_port, out_port, data)

            else:
                out_port = self.get_port(dst_ip)
                if out_port is None:
                    print("shortest--->install_flow: out_port is None in same dp")
                    return
                self.send_flow_mod(first_dp, eth_type, src_ip, dst_ip, in_port, out_port)
                self.send_flow_mod(first_dp, eth_type, dst_ip, src_ip, out_port, in_port)
                self.send_packet_out(first_dp, buffer_id, in_port, out_port, data)

    def install_multicast_flow_test1(self, path, src_switch, dst_switch, eth_type, src_ip, dst_ip, in_port, buffer_id,
                                     data=None):
        """
            下发组播流表
          path nx.graph: steiner tree 路径
        """
        if path is None:
            print("shortest--->install_multicast_flow: Path Error")
            return
        else:
            path = self.dfs_path(path, src_switch)
            print(f"shortest--->install_multicast_flow: [PATH]: {src_switch} <--> {dst_switch}", path)
            first_dp = self.monitor.datapaths_table[src_switch]

            # 将路径转换成一条一条单个路径
            all_path_to_one = self.path_pair_to_one_path(path, src_switch, dst_switch)
            out_ports = []
            # 遍历每个路径
            for _path in all_path_to_one:
                print(f"shortest--->install_multicast_flow _path:", _path)
                # 之后与单播相同
                if len(_path) > 2:
                    print(f"shortest--->install_multicast_flow _path: _path > 2")
                    for i in range(1, len(_path) - 1):
                        port_pair = self.get_port_pair(_path[i - 1], _path[i])
                        port_pair_next = self.get_port_pair(_path[i], _path[i + 1])
                        # print("shortest--->install_multicast_flow: len(path) > 2 port_pair, port_pair_next",
                        #           port_pair, port_pair_next)
                        if port_pair and port_pair_next:
                            src_port, dst_port = port_pair[1], port_pair_next[0]  # 同一个交换机的不同口, 见图
                            datapath = self.monitor.datapaths_table[_path[i]]
                            # 下发正向流表
                            self.send_flow_mod(datapath, eth_type, src_ip, dst_ip, src_port, dst_port)
                        else:
                            print(f"shortestERROR--->install_multicast_flow: len(path) > 2 "
                                  f"path_0, path_1, port_pair: {_path[i - 1], _path[i], port_pair}, "
                                  f"path_1, path_2, next_port_pair: {_path[i], _path[i + 1], port_pair_next}")
                            return

                if len(_path) > 1:
                    # -----------------------------------------------------------------
                    # 最后一个交换机的流表
                    port_pair = self.get_port_pair(_path[-2], _path[-1])
                    if port_pair is None:
                        print("shortest--->install_multicast_flow: port not found")
                        return
                    src_port = port_pair[1]

                    # 通过设定的ip找到主机
                    switch_ip = '10.0.0.' + str(_path[-1])
                    print(f"shortest--->install_multicast_flow: switch_ip == {switch_ip}")
                    dst_port = self.get_port(switch_ip)

                    # {(dpid, in_port): (src_ip, src_mac)}
                    # for switch_ip in setting.DST_MULTICAST_IP[dst_ip]:
                    #     dst_port = self.get_port_by_dpid_ip(_path[-1], switch_ip)

                    if dst_port is None:
                        print("shortest--->install_multicast_flow: Last port is not found")
                        return
                    # 下发最后一个交换机的流表
                    last_dp = self.monitor.datapaths_table[_path[-1]]
                    self.send_flow_mod(last_dp, eth_type, src_ip, dst_ip, src_port, dst_port)
                    # -----------------------------------------------------------------
                    # 第一个交换机的流表
                    port_pair = self.get_port_pair(_path[0], _path[1])
                    if port_pair is None:
                        print("shortest--->install_multicast_flow: port not found in 0 1 switch")
                        return

                    out_port = port_pair[0]
                    # 下发第一个交换机的流表
                    self.send_flow_mod(first_dp, eth_type, src_ip, dst_ip, in_port, out_port)

                    out_ports.append(out_port)

            # 发送packet out
            self.send_multicast_packet_out(first_dp, buffer_id, in_port, out_ports, data)


    def install_multicast_flow(self, path, src_switch, dst_switch, eth_type, src_ip, dst_ip, in_port, buffer_id,
                               data=None):
        """
            下发组播流表
            path nx.graph: steiner tree 路径
        """
        if path is None:
            print("shortest--->install_multicast_flow: Path Error")
            return
        else:
            first_dp = self.monitor.datapaths_table[src_switch]
            # 第一个交换机的out ports
            out_ports = []
            # nodes_father_son_dict = self.drl_multicast.env.remove_redundant_node()
            nodes_father_son_dict = self.get_father_son_switch(path, source=src_switch)
            print("shortest--->install_multicast_flow: ==nodes_father_son_dict ", nodes_father_son_dict)
            # {curr: [father, [son, ]]}
            for curr_node in nodes_father_son_dict.keys():
                father, son = nodes_father_son_dict[curr_node]
                # 如果有子节点说明是中间节点

                if father and son:
                    port_pair = self.get_port_pair(father, curr_node)
                    next_port_pairs = []
                    for s_node in son:
                        next_port_pairs.append(self.get_port_pair(curr_node, s_node))

                    if port_pair and next_port_pairs:
                        src_port = port_pair[1]  # 入口
                        dst_ports = [port[0] for port in next_port_pairs]  # 出口

                        # 如果当前节点是有父子的目的节点
                        if curr_node in dst_switch:
                            # 通过设定的ip找到主机
                            switch_ip = '10.0.0.' + str(curr_node)
                            print(f"shortest--->install_multicast_flow: switch_ip == {switch_ip}")
                            dst_port = self.get_port(switch_ip)
                            dst_ports.append(dst_port)

                        datapath = self.monitor.datapaths_table[curr_node]  # 获取datapath
                        # 下发流表，有多个出口则多个action
                        print(f"shortest--->install_multicast_flow: 下发中间节点 "
                              f"src_port: {src_port}, dst_ports: {dst_ports}")
                        self.send_multicast_flow_mod(datapath, eth_type, src_ip, dst_ip, src_port, dst_ports)
                # 只有父节点，为叶子节点。
                elif father and not son:
                    # 那就从dst_switch里删除这个是叶子节点的node
                    dst_switch.pop(dst_switch.index(curr_node))
                    port_pair = self.get_port_pair(father, curr_node)
                    if port_pair is None:
                        print("shortest--->install_multicast_flow: port not found")
                        return
                    src_port = port_pair[1]

                    # 通过设定的ip找到主机
                    switch_ip = '10.0.0.' + str(curr_node)
                    print(f"shortest--->install_multicast_flow: switch_ip == {switch_ip}")
                    dst_port = self.get_port(switch_ip)

                    if dst_port is None:
                        print("shortest--->install_multicast_flow: Last port is not found")
                        return
                    # 下发最后一个交换机的流表
                    last_dp = self.monitor.datapaths_table[curr_node]
                    self.send_flow_mod(last_dp, eth_type, src_ip, dst_ip, src_port, dst_port)
                    # -----------------------------------------------------------------
                # 只有子节点，为根节点
                elif son and not father:
                    # 第一个交换机的流表
                    for s_node in son:
                        port_pair = self.get_port_pair(src_switch, s_node)

                        if port_pair is None:
                            print(f"shortest--->install_multicast_flow: port not found in {src_switch} {s_node} switch")
                            return
                        out_ports.append(port_pair[0])
                    # 下发第一个交换机的流表
                    self.send_multicast_flow_mod(first_dp, eth_type, src_ip, dst_ip, in_port, out_ports)
            # 发送packet out
            self.send_multicast_packet_out(first_dp, buffer_id, in_port, out_ports, data)

    def send_flow_mod(self, datapath, eth_type, src_ip, dst_ip, src_port, dst_port):
        """
            下发流表
        """
        parser = datapath.ofproto_parser

        actions = [parser.OFPActionOutput(dst_port)]

        match = parser.OFPMatch(in_port=src_port, eth_type=eth_type,
                                ipv4_src=src_ip, ipv4_dst=dst_ip)

        self.add_flow(datapath, 1, match, actions, idle_timeout=15, hard_timeout=60)

    def send_multicast_flow_mod(self, datapath, eth_type, src_ip, dst_ip, src_port, dst_ports):
        """
            下发多个action的流表
        """
        parser = datapath.ofproto_parser

        actions = [parser.OFPActionOutput(dst_port) for dst_port in dst_ports]

        match = parser.OFPMatch(in_port=src_port, eth_type=eth_type,
                                ipv4_src=src_ip, ipv4_dst=dst_ip)

        self.add_flow(datapath, 1, match, actions, idle_timeout=15, hard_timeout=60)

    def add_flow(self, datapath, priority, match, actions, idle_timeout=0, hard_timeout=0):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority, match=match, instructions=inst,
                                idle_timeout=idle_timeout, hard_timeout=hard_timeout)
        datapath.send_msg(mod)

    def send_packet_out(self, datapath, buffer_id, src_port, dst_port, data):
        out = self._build_packet_out(datapath, buffer_id, src_port, dst_port, data)
        if out:
            datapath.send_msg(out)

    def send_multicast_packet_out(self, datapath, buffer_id, src_port, dst_ports, data):
        out = self.build_multicast_packet_out(datapath, buffer_id, src_port, dst_ports, data)
        if out:
            datapath.send_msg(out)

    # 构造输出的包
    @staticmethod
    def build_multicast_packet_out(datapath, buffer_id, src_port, dst_ports, data):
        """ 构造输出的包"""
        actions = []
        for port in dst_ports:
            actions.append(datapath.ofproto_parser.OFPActionOutput(port))

        msg_data = None
        if buffer_id == datapath.ofproto.OFP_NO_BUFFER:
            if data is None:
                return None
            msg_data = data

        out = datapath.ofproto_parser.OFPPacketOut(datapath=datapath, buffer_id=buffer_id,
                                                   data=msg_data, in_port=src_port, actions=actions)

        return out

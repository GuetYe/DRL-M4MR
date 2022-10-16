# network_structure.py
import copy
import time
import xml.etree.ElementTree as ET

from ryu.base import app_manager
from ryu.ofproto import ofproto_v1_3
from ryu.controller import ofp_event
from ryu.controller.handler import set_ev_cls, MAIN_DISPATCHER, CONFIG_DISPATCHER
from ryu.lib import hub
from ryu.lib import igmplib, mac
from ryu.lib.dpid import str_to_dpid
from ryu.lib.packet import packet, arp, ethernet, ipv4, igmp
from ryu.topology import event
from ryu.topology.api import get_switch, get_link, get_host

import networkx as nx
import matplotlib.pyplot as plt

import setting
from setting import print_pretty_table, print_pretty_list


class NetworkStructure(app_manager.RyuApp):
    """
    发现网络拓扑，保存网络结构
    """
    OFP_VERSION = [ofproto_v1_3.OFP_VERSION]

    # _CONTEXTS = {'igmplib': igmplib.IgmpLib}

    def __init__(self, *args, **kwargs):
        super(NetworkStructure, self).__init__(*args, **kwargs)
        self.start_time = time.time()
        self.name = 'discovery'
        # self._snoop = kwargs['igmplib']
        # self._snoop.set_querier_mode(dpid=str_to_dpid('000000000000001e'), server_port=2)
        self.topology_api_app = self
        self.link_info_xml = setting.LINKS_INFO  # xml file path of links info
        self.m_graph = self.parse_topo_links_info()  # 解析mininet构建的topo链路信息

        self.graph = nx.Graph()
        self.pre_graph = nx.Graph()

        self.access_table = {}  # {(dpid, in_port): (src_ip, src_mac)}
        self.switch_all_ports_table = {}  # {dpid: {port_no, ...}}
        self.all_switches_dpid = {}  # dict_key[dpid]
        self.switch_port_table = {}  # {dpid: {port, ...}
        self.link_port_table = {}  # {(src.dpid, dst.dpid): (src.port_no, dst.port_no)}
        self.not_use_ports = {}  # {dpid: {port, ...}}  交换机之间没有用来连接的port
        self.shortest_path_table = {}  # {(src.dpid, dst.dpid): [path]}
        self.arp_table = {}  # {(dpid, eth_src, arp_dst_ip): in_port}
        self.arp_src_dst_ip_table = {}

        # self.multiple_access_table = {}  # {(dpid, in_port): (src_ip, src_mac)}
        # self.group_table = {}  # {group_address: [(dpid, in_port), ...]}

        # self._discover_thread = hub.spawn(self._discover_network_structure)
        # self._show_graph = hub.spawn(self.show_graph_plt())
        self.initiation_delay = setting.INIT_TIME
        self.first_flag = True
        self.cal_path_flag = False

        self._structure_thread = hub.spawn(self.scheduler)
        self._shortest_path_thread = hub.spawn(self.cal_shortest_path_thread)

    def print_parameters(self):
        # self.logger.info("discovery---> access_table: %s", self.access_table)
        # self.logger.info("discovery---> link_port_table: %s", self.link_port_table)
        # self.logger.info("discovery---> not_use_ports: %s", self.not_use_ports)
        # self.logger.info("discovery---> shortest_path_table: %s", self.shortest_path_table)
        logger = self.logger.info if setting.LOGGER else print
        # 图
        # logger("============================= SSSS graph edges==============================")
        # logger('SSSS---> graph edges：\n', self.graph.edges)
        # logger("=============================end SSSS graph edges=============================")

        # 交换机dpid: {交换机所有port号}
        # {dpid: {port_no, ...}}
        print_pretty_table(self.switch_all_ports_table, ['dpid', 'port_no'], [10, 10],
                           'SSSS switch_all_ports_table', logger)

        # 交换机id: lldp发现的端口
        # {dpid: {port, ...}
        print_pretty_table(self.switch_port_table, ['dpid', 'port_no'], [10, 10], 'SSSS switch_port_table',
                           logger)

        # {(dpid, in_port): (src_ip, src_mac)}
        print_pretty_table(self.access_table, ['(dpid, in_port)', '(src_ip, src_mac)'], [10, 40], 'SSSS access_table',
                           logger)

        # {dpid: {port, ...}}
        print_pretty_table(self.not_use_ports, ['dpid', 'not_use_ports'], [10, 30], 'SSSS not_use_ports', logger)

    def scheduler(self):
        i = 0
        while True:
            if i == 3:
                self.get_topology(None)
                i = 0
            hub.sleep(setting.DISCOVERY_PERIOD)
            if setting.PRINT_SHOW:
                self.print_parameters()
            i += 1

    def cal_shortest_path_thread(self):
        self.cal_path_flag = False
        while True:
            if self.cal_path_flag:
                self.calculate_all_nodes_shortest_paths(weight=setting.WEIGHT)
                # print("*****discovery---> self.shortest_path_table:\n", self.shortest_path_table)
            hub.sleep(setting.DISCOVERY_PERIOD)

    # Flow mod and Table miss
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        self.logger.info("discovery---> switch: %s connected", datapath.id)

        # install table miss flow entry
        match = parser.OFPMatch()  # match all
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]

        self.add_flow(datapath, 0, match, actions)

    def add_flow(self, datapath, priority, match, actions):
        inst = [datapath.ofproto_parser.OFPInstructionActions(datapath.ofproto.OFPIT_APPLY_ACTIONS,
                                                              actions)]
        mod = datapath.ofproto_parser.OFPFlowMod(datapath=datapath, priority=priority,
                                                 match=match, instructions=inst)
        datapath.send_msg(mod)

    # Packet In
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        # print("discovery---> discovery PacketIn")
        msg = ev.msg
        datapath = msg.datapath

        # 输入端口号
        in_port = msg.match['in_port']
        pkt = packet.Packet(msg.data)
        arp_pkt = pkt.get_protocol(arp.arp)

        if isinstance(arp_pkt, arp.arp):
            # print("SSSS---> _packet_in_handler: arp packet")
            arp_src_ip = arp_pkt.src_ip
            src_mac = arp_pkt.src_mac
            self.storage_access_info(datapath.id, in_port, arp_src_ip, src_mac)
            # print("discovery--->  access_table:\n    ", self.access_table)

    # 将packet-in解析的arp的网络通路信息存储
    def storage_access_info(self, dpid, in_port, src_ip, src_mac):
        # print(f"SSSS--->storage_access_info, self.access_table: {self.access_table}")
        if in_port in self.not_use_ports[dpid]:
            # print("discovery--->", dpid, in_port, src_ip, src_mac)
            if (dpid, in_port) in self.access_table:
                if self.access_table[(dpid, in_port)] == (src_ip, src_mac):
                    return
                else:
                    self.access_table[(dpid, in_port)] = (src_ip, src_mac)
                    return
            else:
                self.access_table.setdefault((dpid, in_port), None)
                self.access_table[(dpid, in_port)] = (src_ip, src_mac)
                return

    # 利用topology库获取拓扑信息
    events = [event.EventSwitchEnter, event.EventSwitchLeave,
              event.EventPortAdd, event.EventPortDelete, event.EventPortModify,
              event.EventLinkAdd, event.EventLinkDelete]

    @set_ev_cls(events)
    def get_topology(self, ev):
        present_time = time.time()
        if present_time - self.start_time < self.initiation_delay:  # Set to 30s
            print(f'SSSS--->get_topology: need to WAIT {self.initiation_delay - (present_time - self.start_time):.2f}s')
            return
        elif self.first_flag:
            self.first_flag = False
            print("SSSS--->get_topology: complete WAIT")

        # self.logger.info("discovery---> EventSwitch/Port/Link")
        self.logger.info("[Topology Discovery Ok]")
        # 事件发生时，获得swicth列表
        switch_list = get_switch(self.topology_api_app, None)
        # 将swicth添加到self.switch_all_ports_table
        for switch in switch_list:
            dpid = switch.dp.id
            self.switch_all_ports_table.setdefault(dpid, set())
            self.switch_port_table.setdefault(dpid, set())
            self.not_use_ports.setdefault(dpid, set())
            # print("discovery---> ",switch, switch.ports)
            for p in switch.ports:
                self.switch_all_ports_table[dpid].add(p.port_no)

        self.all_switches_dpid = self.switch_all_ports_table.keys()

        # time.sleep(0.5)
        # 获得link
        link_list = get_link(self.topology_api_app, None)
        # print("discovery---> ",len(link_list))

        # 将link添加到self.link_table
        for link in link_list:
            src = link.src  # 实际是个port实例，我找了半天
            dst = link.dst
            self.link_port_table[(src.dpid, dst.dpid)] = (src.port_no, dst.port_no)

            if src.dpid in self.all_switches_dpid:
                self.switch_port_table[src.dpid].add(src.port_no)
            if dst.dpid in self.all_switches_dpid:
                self.switch_port_table[dst.dpid].add(dst.port_no)

        # 统计没使用的端口
        for sw_dpid in self.switch_all_ports_table.keys():
            all_ports = self.switch_all_ports_table[sw_dpid]
            linked_port = self.switch_port_table[sw_dpid]
            # print("discovery---> all_ports, linked_port", all_ports, linked_port)
            self.not_use_ports[sw_dpid] = all_ports - linked_port

        # 建立拓扑 bw和delay未定
        self.build_topology_between_switches()
        self.cal_path_flag = True

    def build_topology_between_switches(self, bw=0, delay=0, loss=0):
        """ 根据 src_dpid 和 dst_dpid 建立拓扑，bw 和 delay 信息还未定"""
        # networkxs使用已有Link的src_dpid和dst_dpid信息建立拓扑
        _graph = nx.Graph()

        # self.graph.clear()
        for (src_dpid, dst_dpid) in self.link_port_table.keys():
            # 建立switch之间的连接，端口可以通过查link_port_table获得
            _graph.add_edge(src_dpid, dst_dpid, bw=bw, delay=delay, loss=loss)
        if _graph.edges == self.graph.edges:
            return 
        else:
            self.graph = _graph

    def calculate_weight(self, node1, node2, weight_dict):
        """ 计算路径时，weight可以调用函数，该函数根据因子计算 bw * factor - delay * (1 - factor) 后的weight"""
        # weight可以调用的函数
        assert 'bw' in weight_dict and 'delay' in weight_dict, "edge weight should have bw and delay"
        try:
            weight = weight_dict['bw'] * setting.FACTOR - weight_dict['delay'] * (1 - setting.FACTOR)
            return weight
        except TypeError:
            print("discovery ERROR---> weight_dict['bw']: ", weight_dict['bw'])
            print("discovery ERROR---> weight_dict['delay']: ", weight_dict['delay'])
            return None

    def get_shortest_paths(self, src_dpid, dst_dpid, weight=None):
        """ 计算src到dst的最短路径，存在self.shortest_path_table中"""
        graph = self.graph.copy()
        # print(graph.edges)
        # print("SSSS--->get_shortest_paths ==calculate shortest path %s to %s" % (src_dpid, dst_dpid))
        self.shortest_path_table[(src_dpid, dst_dpid)] = nx.shortest_path(graph,
                                                                          source=src_dpid,
                                                                          target=dst_dpid,
                                                                          weight=weight,
                                                                          method=setting.METHOD)
        # print("SSSS--->get_shortest_paths ==[PATH] %s <---> %s: %s" % (
        #     src_dpid, dst_dpid, self.shortest_path_table[(src_dpid, dst_dpid)]))

    def calculate_all_nodes_shortest_paths(self, weight=None):
        """ 根据已构建的图，计算所有nodes间的最短路径，weight为权值，可以为calculate_weight()该函数"""
        self.shortest_path_table = {}  # 先清空，再计算
        for src in self.graph.nodes():
            for dst in self.graph.nodes():
                if src != dst:
                    self.get_shortest_paths(src, dst, weight=weight)
                else:
                    continue

    def get_host_ip_location(self, host_ip):
        """ 
            通过host_ip查询 self.access_table: {(dpid, in_port): (src_ip, src_mac)}
            获得(dpid, in_port)
        """
        if host_ip == "0.0.0.0" or host_ip == "255.255.255.255":
            return None

        for key in self.access_table.keys():  # {(dpid, in_port): (src_ip, src_mac)}
            if self.access_table[key][0] == host_ip:
                # print("discovery--->zzzz---> key", key)
                return key
        print("SSS--->get_host_ip_location: %s location is not found" % host_ip)
        return None

    def get_ip_by_dpid(self, dpid):
        """
            通过 dpid 查询  {(dpid, in_port): (src_ip, src_mac)}
            获得 ip  src_ip
        """
        for key, value in self.access_table.items():
            if key[0] == dpid:
                return value[0]
        print("SSS--->get_ip_by_dpid: %s ip is not found" % dpid)
        return None

    def parse_topo_links_info(self):
        m_graph = nx.Graph()
        parser = ET.parse(self.link_info_xml)
        root = parser.getroot()

        # links_info_element = root.find("links_info")

        def _str_tuple2int_list(s: str):
            s = s.strip()
            assert s.startswith('(') and s.endswith(")"), '应该为str的元组，如 "(1, 2)"'
            s_ = s[1: -1].split(', ')
            return [int(i) for i in s_]

        node1, node2, port1, port2, bw, delay, loss = None, None, None, None, None, None, None
        for e in root.iter():
            if e.tag == 'links':
                node1, node2 = _str_tuple2int_list(e.text)
            elif e.tag == 'ports':
                port1, port2 = _str_tuple2int_list(e.text)
            elif e.tag == 'bw':
                bw = float(e.text)
            elif e.tag == 'delay':
                delay = float(e.text[:-2])
            elif e.tag == 'loss':
                loss = float(e.text)
            else:
                print(e.tag)
                continue
            m_graph.add_edge(node1, node2, port1=port1, port2=port2, bw=bw, delay=delay, loss=loss)

        for edge in m_graph.edges(data=True):
            print(edge)
        return m_graph

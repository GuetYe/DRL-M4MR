# -*- coding: utf-8 -*-
# @File    : network_monitor.py
# @Date    : 2021-08-12
# @Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来-
# network_monitor.py
import copy
from operator import attrgetter

from ryu.base import app_manager
from ryu.ofproto import ofproto_v1_3
from ryu.controller import ofp_event
from ryu.controller.handler import set_ev_cls, MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.lib import hub
from ryu.base.app_manager import lookup_service_brick

import setting
from setting import print_pretty_table, print_pretty_list


class NetworkMonitor(app_manager.RyuApp):
    """ 监控网络流量状态"""
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(NetworkMonitor, self).__init__(*args, **kwargs)
        self.name = 'monitor'
        # {dpid: datapath}
        self.datapaths_table = {}
        # {dpid:{port_no: (config, state, curr_speed, max_speed)}}
        self.dpid_port_fueatures_table = {}
        # {(dpid, port_no): (stat.tx_bytes, stat.rx_bytes, stat.rx_errors, stat.duration_sec,
        # stat.duration_nsec, stat.tx_packets, stat.rx_packets)}
        self.port_stats_table = {}
        # {dpid:{(in_port, ipv4_dsts, out_port): (packet_count, byte_count, duration_sec, duration_nsec)}}
        self.flow_stats_table = {}
        # {(dpid, port_no): [speed, .....]}
        self.port_speed_table = {}
        # {dpid: {(in_port, ipv4_dsts, out_port): speed}}
        self.flow_speed_table = {}

        self.port_flow_dpid_stats = {'port': {}, 'flow': {}}
        # {dpid: {port_no: curr_bw}}
        self.port_curr_speed = {}

        self.port_loss = {}

        self.discovery = lookup_service_brick("discovery")  # 创建一个NetworkStructure的实例

        # self.monitor_thread = hub.spawn(self._monitor)
        self.monitor_thread = hub.spawn(self.scheduler)
        self.save_thread = hub.spawn(self.save_bw_loss_graph)

    def print_parameters(self):
        # print("monitor---> self.datapaths_table", self.datapaths_table)
        # print("monitor---> self.dpid_port_fueatures_table", self.dpid_port_fueatures_table)
        # print("monitor---> self.port_stats_table", self.port_stats_table)
        # print("monitor---> self.flow_stats_table", self.flow_stats_table)
        # print("monitor---> self.port_speed_table", self.port_speed_table)
        # print("monitor---> self.flow_speed_table", self.flow_speed_table)
        # print("monitor---> self.port_curr_speed", self.port_curr_speed)

        logger = self.logger.info if setting.LOGGER else print

        # {dpid: datapath}
        # print_pretty_table(self.datapaths_table, ['dpid', 'datapath'], [6, 64], 'MMMM datapaths_table',
        #                    logger)

        # # {dpid:{port_no: (config, state, curr_speed, max_speed)}}
        # print_pretty_table(self.dpid_port_fueatures_table,
        #                    ['dpid', 'port_no:(config, state, curr_speed, max_speed)'],
        #                    [6, 40], 'MMMM dpid_port_fueatures_table', logger)

        # {(dpid, port_no): (stat.tx_bytes, stat.rx_bytes, stat.rx_errors, stat.duration_sec,
        # stat.duration_nsec, stat.tx_packets, stat.rx_packets)}
        # print_pretty_table(self.port_stats_table,
        #                    ['(dpid, port_no)',
        #                     '(stat.tx_bytes, stat.rx_bytes, stat.rx_errors, stat.duration_sec, stat.duration_nsec, stat.tx_packets, stat.rx_packets)'],
        #                    [18, 120], 'MMMM port_stats_table', logger)

        # {(dpid, port_no): [speed, .....]}
        # print_pretty_table(self.port_speed_table,
        #                    ['(dpid, port_no)', 'speed'],
        #                    [18, 40], 'MMMM port_speed_table', logger)

        print("'MMMM port_loss: \n", self.port_loss)

    def print_parameters_(self):
        print("monitor---------- %s ----------", self.name)
        for attr, value in self.__dict__.items():
            print("monitor\n---> %s: %s" % attr, value)
        print("monitor===================================")

    def scheduler(self):
        while True:
            self.port_flow_dpid_stats['flow'] = {}
            self.port_flow_dpid_stats['port'] = {}

            self._request_stats()
            if setting.PRINT_SHOW:
                self.print_parameters()
            hub.sleep(setting.MONITOR_PERIOD)

    def save_bw_loss_graph(self):
        while True:
            self.create_bandwidth_graph()
            self.create_loss_graph()
            hub.sleep(setting.MONITOR_PERIOD)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        """ 存放所有的datapath实例"""
        datapath = ev.datapath  # OFPStateChange类可以直接获得datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths_table:
                print("MMMM--->  register datapath: %016x" % datapath.id)
                self.datapaths_table[datapath.id] = datapath

                # 一些初始化
                self.dpid_port_fueatures_table.setdefault(datapath.id, {})
                self.flow_stats_table.setdefault(datapath.id, {})

        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths_table:
                print("MMMM--->  unreigster datapath: %016x" % datapath.id)
                del self.datapaths_table[datapath.id]

    # 主动发送request，请求状态信息
    def _request_stats(self):
        # print("MMMM--->  send request --->   ---> send request ---> ")
        datapaths_table = self.datapaths_table.values()

        for datapath in list(datapaths_table):
            self.dpid_port_fueatures_table.setdefault(datapath.id, {})
            # print("MMMM--->  send stats request: %016x", datapath.id)
            ofproto = datapath.ofproto
            parser = datapath.ofproto_parser

            # 1. 端口描述请求
            req = parser.OFPPortDescStatsRequest(datapath, 0)  #
            datapath.send_msg(req)

            # 2. 端口统计请求
            req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)  # 所有端口
            datapath.send_msg(req)

            # 3. 单个流统计请求
            # req = parser.OFPFlowStatsRequest(datapath)
            # datapath.send_msg(req)

    # 处理上面请求的回复OFPPortDescStatsReply
    @set_ev_cls(ofp_event.EventOFPPortDescStatsReply, MAIN_DISPATCHER)
    def port_desc_stats_reply_handler(self, ev):
        """ 存储端口描述信息, 见OFPPort类, 配置、状态、当前速度"""
        # print("MMMM--->  EventOFPPortDescStatsReply")
        msg = ev.msg
        dpid = msg.datapath.id
        ofproto = msg.datapath.ofproto

        config_dict = {ofproto.OFPPC_PORT_DOWN: 'Port Down',
                       ofproto.OFPPC_NO_RECV: 'No Recv',
                       ofproto.OFPPC_NO_FWD: 'No Forward',
                       ofproto.OFPPC_NO_PACKET_IN: 'No Pakcet-In'}

        state_dict = {ofproto.OFPPS_LINK_DOWN: "Link Down",
                      ofproto.OFPPS_BLOCKED: "Blocked",
                      ofproto.OFPPS_LIVE: "Live"}

        for ofport in ev.msg.body:  # 这一直有bug，修改properties
            if ofport.port_no != ofproto_v1_3.OFPP_LOCAL:  # 0xfffffffe  4294967294

                if ofport.config in config_dict:
                    config = config_dict[ofport.config]
                else:
                    config = 'Up'

                if ofport.state in state_dict:
                    state = state_dict[ofport.state]
                else:
                    state = 'Up'

                # 存储配置，状态, curr_speed,max_speed=0
                port_features = (config, state, ofport.curr_speed, ofport.max_speed)
                # print("MMMM--->  ofport.curr_speed", ofport.curr_speed)
                self.dpid_port_fueatures_table[dpid][ofport.port_no] = port_features

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def port_stats_table_reply_handler(self, ev):
        """ 存储端口统计信息, 见OFPPortStats, 发送bytes、接收bytes、生效时间duration_sec等
         Replay message content:
            (stat.port_no,
             stat.rx_packets, stat.tx_packets,
             stat.rx_bytes, stat.tx_bytes,
             stat.rx_dropped, stat.tx_dropped,
             stat.rx_errors, stat.tx_errors,
             stat.rx_frame_err, stat.rx_over_err,
             stat.rx_crc_err, stat.collisions,
             stat.duration_sec, stat.duration_nsec))
        """
        # print("MMMM--->  EventOFPPortStatsReply")
        body = ev.msg.body
        dpid = ev.msg.datapath.id
        self.port_flow_dpid_stats['port'][dpid] = body
        # self.port_curr_speed.setdefault(dpid, {})

        for stat in sorted(body, key=attrgetter("port_no")):
            port_no = stat.port_no
            if port_no != ofproto_v1_3.OFPP_LOCAL:
                key = (dpid, port_no)
                value = (stat.tx_bytes, stat.rx_bytes, stat.rx_errors,
                         stat.duration_sec, stat.duration_nsec, stat.tx_packets, stat.rx_packets)
                self._save_stats(self.port_stats_table, key, value, 5)  # 保存信息，最多保存前5次

                pre_bytes = 0
                # delta_time = setting.MONITOR_PERIOD
                delta_time = setting.SCHEDULE_PERIOD
                stats = self.port_stats_table[key]  # 获得已经存了的统计信息

                if len(stats) > 1:  # 有两次以上的信息
                    pre_bytes = stats[-2][0] + stats[-2][1]
                    delta_time = self._calculate_delta_time(stats[-1][3], stats[-1][4],
                                                            stats[-2][3], stats[-2][4])  # 倒数第一个统计信息，倒数第二个统计信息

                speed = self._calculate_speed(stats[-1][0] + stats[-1][1],
                                              pre_bytes, delta_time)
                self._save_stats(self.port_speed_table, key, speed, 5)
                self._calculate_port_speed(dpid, port_no, speed)

        self.calculate_loss_of_link()

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        """ 存储flow的状态，算这个干啥。。"""
        msg = ev.msg
        body = msg.body
        datapath = msg.datapath
        dpid = datapath.id

        self.port_flow_dpid_stats['flow'][dpid] = body
        # print("MMMM--->  body", body)

        for stat in sorted([flowstats for flowstats in body if flowstats.priority == 1],
                           key=lambda flowstats: (flowstats.match.get('in_port'), flowstats.match.get('ipv4_dst'))):
            # print("MMMM--->  stat.match", stat.match)
            # print("MMMM--->  stat", stat)
            key = (stat.match['in_port'], stat.match['ipv4_dst'],
                   stat.instructions[0].actions[0].port)
            value = (stat.packet_count, stat.byte_count, stat.duration_sec, stat.duration_nsec)
            self._save_stats(self.flow_stats_table[dpid], key, value, 5)

            pre_bytes = 0
            # delta_time = setting.MONITOR_PERIOD
            delta_time = setting.SCHEDULE_PERIOD
            value = self.flow_stats_table[dpid][key]
            if len(value) > 1:
                pre_bytes = value[-2][1]
                # print("MMMM--->  _flow_stats_reply_handler delta_time: now", value[-1][2], value[-1][3], "pre",
                #       value[-2][2],
                #       value[-2][3])
                delta_time = self._calculate_delta_time(value[-1][2], value[-1][3],
                                                        value[-2][2], value[-2][3])
            speed = self._calculate_speed(self.flow_stats_table[dpid][key][-1][1], pre_bytes, delta_time)
            self.flow_speed_table.setdefault(dpid, {})
            self._save_stats(self.flow_speed_table[dpid], key, speed, 5)

    # 存多次数据，比如一个端口存上一次的统计信息和这一次的统计信息
    @staticmethod
    def _save_stats(_dict, key, value, keep):
        if key not in _dict:
            _dict[key] = []
        _dict[key].append(value)

        if len(_dict[key]) > keep:
            _dict[key].pop(0)  # 弹出最早的数据

    def _calculate_delta_time(self, now_sec, now_nsec, pre_sec, pre_nsec):
        """ 计算统计时间, 即两个消息时间差"""
        return self._calculate_seconds(now_sec, now_nsec) - self._calculate_seconds(pre_sec, pre_nsec)

    @staticmethod
    def _calculate_seconds(sec, nsec):
        """ 计算 sec + nsec 的和，单位为 seconds"""
        return sec + nsec / 10 ** 9

    @staticmethod
    def _calculate_speed(now_bytes, pre_bytes, delta_time):
        """ 计算统计流量速度"""
        if delta_time:

            return (now_bytes - pre_bytes) / delta_time
        else:
            return 0

    def _calculate_port_speed(self, dpid, port_no, speed):
        curr_bw = speed * 8 / 10 ** 6  # MBit/s
        # print(f"monitorMMMM---> _calculate_port_speed: {curr_bw} MBits/s", )
        self.port_curr_speed.setdefault(dpid, {})
        self.port_curr_speed[dpid][port_no] = curr_bw

    @set_ev_cls(ofp_event.EventOFPPortStatus, MAIN_DISPATCHER)
    def _port_status_handler(self, ev):
        """ 处理端口状态： ADD, DELETE, MODIFIED"""
        msg = ev.msg
        dp = msg.datapath
        ofp = dp.ofproto

        if msg.reason == ofp.OFPPR_ADD:
            reason = 'ADD'
        elif msg.reason == ofp.OFPPR_DELETE:
            reason = 'DELETE'
        elif msg.reason == ofp.OFPPR_MODIFY:
            reason = 'MODIFY'
        else:
            reason = 'unknown'

        print('MMMM---> _port_status_handler OFPPortStatus received: reason=%s desc=%s' % (reason, msg.desc))

    # 通过获得的网络拓扑，更新其bw权重
    def create_bandwidth_graph(self):
        # print("MMMM--->  create bandwidth graph")
        for link in self.discovery.link_port_table:
            src_dpid, dst_dpid = link
            src_port, dst_port = self.discovery.link_port_table[link]

            if src_dpid in self.port_curr_speed.keys() and dst_dpid in self.port_curr_speed.keys():
                src_port_bw = self.port_curr_speed[src_dpid][src_port]
                dst_port_bw = self.port_curr_speed[dst_dpid][dst_port]
                src_dst_bandwidth = min(src_port_bw, dst_port_bw)  # bottleneck bandwidth

                # print(f"monitor--> dst[{src_dpid}]_port[{src_port}]_bw:  %.5f" % dst_port_bw)
                # print(f"monitor---> src[{dst_dpid}]_port[{dst_port}]_bw:  %.5f" % src_port_bw)
                # print("monitor---> src_dst_bandwidth:   %.5f" % src_dst_bandwidth)

                # 对图的edge设置 可用bw 值
                capacity = self.discovery.m_graph[src_dpid][dst_dpid]['bw']
                self.discovery.graph[src_dpid][dst_dpid]['bw'] = max(capacity - src_dst_bandwidth, 0)

            else:
                self.logger.info(
                    "MMMM--->  create_bandwidth_graph: [{}] [{}] not in port_free_bandwidth ".format(src_dpid,
                                                                                                     dst_dpid))
                self.discovery.graph[src_dpid][dst_dpid]['bw'] = -1

        # print("MMMM--->  ", self.discovery.graph.edges(data=True))
        # print("MMMM---> " * 2, self.discovery.count + 1)

    # calculate loss tx - rx / tx
    def calculate_loss_of_link(self):
        """
            发端口 和 收端口 ，端口loss
        """
        for link, port in self.discovery.link_port_table.items():
            src_dpid, dst_dpid = link
            src_port, dst_port = port
            if (src_dpid, src_port) in self.port_stats_table.keys() and \
                    (dst_dpid, dst_port) in self.port_stats_table.keys():
                # {(dpid, port_no): (stat.tx_bytes, stat.rx_bytes, stat.rx_errors, stat.duration_sec,
                # stat.duration_nsec, stat.tx_packets, stat.rx_packets)}
                # 1. 顺向  2022/3/11 packets modify--> bytes
                tx = self.port_stats_table[(src_dpid, src_port)][-1][0]  # tx_bytes
                rx = self.port_stats_table[(dst_dpid, dst_port)][-1][1]  # rx_bytes
                loss_ratio = abs(float(tx - rx) / tx) * 100
                self._save_stats(self.port_loss, link, loss_ratio, 5)
                # print(f"MMMM--->[{link}]({dst_dpid}, {dst_port}) rx: ", rx, "tx: ", tx,
                #       "loss_ratio: ", loss_ratio)

                # 2. 逆项
                tx = self.port_stats_table[(dst_dpid, dst_port)][-1][0]  # tx_bytes
                rx = self.port_stats_table[(src_dpid, src_port)][-1][1]  # rx_bytes
                loss_ratio = abs(float(tx - rx) / tx) * 100
                self._save_stats(self.port_loss, link[::-1], loss_ratio, 5)

                # print(f"MMMM--->[{link[::-1]}]({dst_dpid}, {dst_port}) rx: ", rx, "tx: ", tx,
                #       "loss_ratio: ", loss_ratio)
            else:
                self.logger.info("MMMM--->  calculate_loss_of_link error", )

    # update graph loss
    def update_graph_loss(self):
        """从1 往2 和 从2 往1，取最大作为链路loss """
        for link in self.discovery.link_port_table:
            src_dpid = link[0]
            dst_dpid = link[1]
            if link in self.port_loss.keys() and link[::-1] in self.port_loss.keys():
                src_loss = self.port_loss[link][-1]  # 1-->2  -1取最新的那个
                dst_loss = self.port_loss[link[::-1]][-1]  # 2-->1
                link_loss = max(src_loss, dst_loss)  # 百分比 max loss between port1 and port2
                self.discovery.graph[src_dpid][dst_dpid]['loss'] = link_loss

                # print(f"MMMM---> update_graph_loss link[{link}]_loss: ", link_loss)
            else:
                self.discovery.graph[src_dpid][dst_dpid]['loss'] = 100

    def create_loss_graph(self):
        """
            在graph中更新边的loss值
        """
        # self.calculate_loss_of_link()
        self.update_graph_loss()

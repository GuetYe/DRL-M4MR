# -*- coding: utf-8 -*-
# @File    : arp_handler.py
# @Date    : 2022-03-09
# @Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来-
from ryu.base import app_manager
from ryu.base.app_manager import lookup_service_brick
from ryu.ofproto import ofproto_v1_3
from ryu.controller import ofp_event
from ryu.controller.handler import set_ev_cls, MAIN_DISPATCHER
from ryu.lib.packet import packet
from ryu.lib.packet import arp, ipv4, ethernet

ETHERNET = ethernet.ethernet.__name__
ETHERNET_MULTICAST = "ff:ff:ff:ff:ff:ff"
ARP = arp.arp.__name__


class ArpHandler(app_manager.RyuApp):
    OFP_VERSION = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(ArpHandler, self).__init__(*args, **kwargs)
        self.discovery = lookup_service_brick('discovery')
        self.monitor = lookup_service_brick('monitor')

        self.arp_table = {}
        self.sw = {}
        self.mac_to_port = {}

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        """
            处理PacketIn事件
            1. arp包 是否已经记录
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

        eth = pkt.get_protocols(ethernet.ethernet)[0]
        src = eth.src

        header_list = dict((p.protocol_name, p) for p in pkt.protocols if type(p) != bytes)
        # print("shortest--->_packet_in_handler: header_list:\n    ", header_list)
        if isinstance(arp_pkt, arp.arp):
            self.arp_table[arp_pkt.src_ip] = src

            if self.arp_handler(header_list, datapath, in_port, msg.buffer_id):
                # 1:reply or drop;  0: flood
                # print("ARP_PROXY_13")
                return None
            else:
                arp_src_ip = arp_pkt.src_ip
                arp_dst_ip = arp_pkt.dst_ip
                location = self.discovery.get_host_ip_location(arp_dst_ip)
                # print("shortest--->zzzzzz----> location", location)
                if location:  # 如果有这个主机的位置
                    # print("shortest--->_packet_in_handler: ==Reply Arp to knew host")
                    dpid_dst, out_port = location
                    datapath = self.monitor.datapaths_table[dpid_dst]
                    out = self._build_packet_out(datapath, ofproto.OFP_NO_BUFFER, ofproto.OFPP_CONTROLLER,
                                                 out_port, msg.data)
                    datapath.send_msg(out)
                    return
                else:
                    print("shortest--->_packet_in_handler: ==Flooding")
                    for dpid in self.discovery.switch_all_ports_table:
                        for port in self.discovery.switch_all_ports_table[dpid]:
                            if (dpid, port) not in self.discovery.access_table.keys():  # 如果不知道
                                datapath = self.monitor.datapaths_table[dpid]
                                out = self._build_packet_out(datapath, ofproto.OFP_NO_BUFFER,
                                                             ofproto.OFPP_CONTROLLER, port, msg.data)
                                datapath.send_msg(out)
                    return

    def arp_handler(self, header_list, datapath, in_port, msg_buffer_id):
        header_list = header_list
        datapath = datapath
        in_port = in_port

        # if ETHERNET in header_list:
        eth_dst = header_list[ETHERNET].dst
        eth_src = header_list[ETHERNET].src

        # print("shortest---> arp_handler eth_dst eth_src: \n", eth_dst, eth_src)

        if eth_dst == ETHERNET_MULTICAST and ARP in header_list:
            arp_dst_ip = header_list[ARP].dst_ip
            if (datapath.id, eth_src, arp_dst_ip) in self.sw:  # break loop
                # print("shortest---> arp_handler: ====BREAK LOOP")
                out = datapath.ofproto_parser.OFPPacketOut(
                    datapath=datapath,
                    buffer_id=datapath.ofproto.OFP_NO_BUFFER,
                    in_port=in_port,
                    actions=[], data=None
                )
                datapath.send_msg(out)
                return True
            else:
                self.sw[(datapath.id, eth_src, arp_dst_ip)] = in_port

        if ARP in header_list:
            # print("discovery---> arp_handler: ====ARP ARP")
            hwtype = header_list[ARP].hwtype
            proto = header_list[ARP].proto
            hlen = header_list[ARP].hlen
            plen = header_list[ARP].plen
            opcode = header_list[ARP].opcode

            arp_src_ip = header_list[ARP].src_ip
            arp_dst_ip = header_list[ARP].dst_ip

            actions = []

            if opcode == arp.ARP_REQUEST:
                if arp_dst_ip in self.arp_table:  # arp reply
                    # print("shortest---> arp_handler: ====ARP REPLY")
                    actions.append(datapath.ofproto_parser.OFPActionOutput(in_port))

                    ARP_Reply = packet.Packet()
                    ARP_Reply.add_protocol(ethernet.ethernet(ethertype=header_list[ETHERNET].ethertype,
                                                             dst=eth_src,
                                                             src=self.arp_table[arp_dst_ip]))
                    ARP_Reply.add_protocol(arp.arp(opcode=arp.ARP_REPLY,
                                                   src_mac=self.arp_table[arp_dst_ip],
                                                   src_ip=arp_dst_ip,
                                                   dst_mac=eth_src,
                                                   dst_ip=arp_src_ip))

                    ARP_Reply.serialize()

                    out = datapath.ofproto_parser.OFPPacketOut(
                        datapath=datapath,
                        buffer_id=datapath.ofproto.OFP_NO_BUFFER,
                        in_port=datapath.ofproto.OFPP_CONTROLLER,
                        actions=actions,
                        data=ARP_Reply.data
                    )
                    datapath.send_msg(out)
                    return True
        return False

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

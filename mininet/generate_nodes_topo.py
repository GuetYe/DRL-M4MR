# -*- coding: utf-8 -*-
# @File    : GEANT_23nodes_topo.py.py
# @Date    : 2021-12-09
# @Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来-
import os
import random
from pathlib import Path
import time
import json
import threading
import xml.etree.ElementTree as ET

import networkx

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel
from mininet.util import dumpNodeConnections

random.seed(2020)


def generate_port(node_idx1, node_idx2):
    if (node_idx2 > 9) and (node_idx1 > 9):
        port = str(node_idx1) + "0" + str(node_idx2)
    else:
        port = str(node_idx1) + "00" + str(node_idx2)  # test

    return int(port)


def generate_switch_port(graph):
    switch_port_dict = {}
    for node in graph.nodes:
        switch_port_dict.setdefault(node, list(range(graph.degree[node])))
    return switch_port_dict


def parse_xml_topology(topology_path):
    """
        parse topology from topology.xml
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


def create_topo_links_info_xml(path, links_info):
    """
        <links_info>
            <links> (switch1, switch2)
                <ports>(1, 1)</ports>
                <bw>100</bw>
                <delay>5ms</delay>
                <loss>1</loss>
            </links>
        </links_info>
    :param path: 保存路径
    :param links_info: 链路信息字典 {link: {ports, bw, delay, loss}}
    :return: None
    """
    # 根节点
    root = ET.Element('links_info')

    for link, info in links_info.items():
        # 子节点
        child = ET.SubElement(root, 'links')
        child.text = str(link)

        # 二级子节点
        sub_child1 = ET.SubElement(child, 'ports')
        sub_child1.text = str((info['port1'], info['port2']))

        sub_child2 = ET.SubElement(child, 'bw')
        sub_child2.text = str(info['bw'])

        sub_child2 = ET.SubElement(child, 'delay')
        sub_child2.text = str(info['delay'])

        sub_child2 = ET.SubElement(child, 'loss')
        sub_child2.text = str(info['loss'])

    tree = ET.ElementTree(root)
    Path(path).parent.mkdir(exist_ok=True)
    tree.write(path, encoding='utf-8', xml_declaration=True)
    print('saved links info xml.')


def get_mininet_device(net, idx: list, device='h'):
    """
        获得idx中mininet的实例, 如 h1, h2 ...;  s1, s2 ...
    :param net: mininet网络实例
    :param idx: 设备标号集合, list
    :param device: 设备名称 'h', 's'
    :return d: dict{idx: 设备mininet实例}
    """
    d = {}
    for i in idx:
        d.setdefault(i, net.get(f'{device}{i}'))

    return d


def run_corresponding_sh_script(devices: dict, label_path):
    """
        对应的host运行对应的shell脚本
    :param devices: {idx: device}
    :param label_path:  './24nodes/TM-{}/{}/{}_'
    """
    p = label_path + '{}.sh'
    for i, d in devices.items():
        if i < 9:
            i = f'0{i}'
        else:
            i = f'{i}'
        p = p.format(i)
        _cmd = f'bash {p}'
        d.cmd(_cmd)
    print(f"---> complete run {label_path}")


def run_ip_add_default(hosts: dict):
    """
        运行 ip route add default via 10.0.0.x 命令
    """
    _cmd = 'ip route add default via 10.0.0.'
    for i, h in hosts.items():
        print(_cmd + str(i))
        h.cmd(_cmd + str(i))
    print("---> run ip add default complete")


def _test_cmd(devices: dict, my_cmd):
    for i, d in devices.items():
        d.cmd(my_cmd)
        print(f'exec {my_cmd}zzz{i}')
        # print(f'return {r}')


def run_iperf(path, host):
    _cmd = 'bash ' + path + '&'
    host.cmd(_cmd)


def all_host_run_iperf(hosts: dict, path, finish_file):
    """
        path = r'./iperfTM/'
    """
    idxs = len(os.listdir(path))
    path = path + '/TM-'
    for idx in range(idxs):
        script_path = path + str(idx)
        for i, h in hosts.items():
            servers_cmd = script_path + '/Servers/server_' + str(i) + '.sh'
            _cmd = 'bash '
            print(_cmd + servers_cmd)
            h.cmd(_cmd + servers_cmd)

        for i, h in hosts.items():
            clients_cmd = script_path + '/Clients/client_' + str(i) + '.sh'
            _cmd = 'bash '
            print(_cmd + clients_cmd + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            h.cmd(_cmd + clients_cmd)

        time.sleep(300)

    write_iperf_time(finish_file)


def write_pinall_time(finish_file):
    with open(finish_file, "w+") as f:
        _content = {
            "ping_all_finish_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "start_save_flag": True,
            "finish_flag": False
        }
        json.dump(_content, f)


def write_iperf_time(finish_file):
    with open(finish_file, "r+") as f:
        _read = json.load(f)
        _content = {
            "iperf_finish_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "finish_flag": True,
        }
        _read.update(_content)

    with open(finish_file, "w+") as f:
        json.dump(_read, f)


def remove_finish_file(finish_file):
    try:
        os.remove(finish_file)
    except FileNotFoundError:
        pass


def net_h1_ping_others(net):
    hosts = net.hosts
    for h in hosts[1:]:
        net.ping((hosts[0], h))


class GEANT23nodesTopo(Topo):
    def __init__(self, graph):
        super(GEANT23nodesTopo, self).__init__()
        self.node_idx = graph.nodes
        self.edges_pairs = graph.edges
        self.bw1 = 100  # Gbps -> M
        self.bw2 = 25  # Gbps -> M
        self.bw3 = 1.15  # Mbps
        self.bw4 = 100  # host -- switch
        self.delay = 20
        self.loss = 10

        self.host_port = 9
        self.snooper_port = 10

        self.bw1_links = [(12, 22), (12, 10), (12, 2), (13, 17), (4, 2), (4, 16), (1, 3), (1, 7), (1, 16), (3, 10),
                          (3, 21), (10, 16), (10, 17), (7, 17), (7, 2), (7, 21), (16, 9), (20, 17)]
        self.bw2_links = [(13, 19), (13, 2), (19, 7), (23, 17), (23, 2), (8, 5), (8, 9), (18, 2), (18, 21), (5, 16),
                          (3, 11), (10, 11), (22, 20), (20, 15), (9, 15)]
        self.bw3_links = [(13, 14), (19, 6), (3, 14), (7, 6)]

        def _return_bw(link: tuple):
            if link in self.bw1_links:
                return self.bw1
            elif link in self.bw2_links:
                return self.bw2
            elif link in self.bw3_links:
                return self.bw3
            else:
                raise ValueError

        # 添加交换机
        switches = {}
        for s in self.node_idx:
            switches.setdefault(s, self.addSwitch('s{0}'.format(s)))
            print('添加交换机:', s)

        switch_port_dict = generate_switch_port(graph)
        links_info = {}
        # 添加链路
        for l in self.edges_pairs:
            port1 = switch_port_dict[l[0]].pop(0) + 1
            port2 = switch_port_dict[l[1]].pop(0) + 1
            bw = _return_bw(l)

            _d = str(random.randint(0, self.delay)) + 'ms'
            _l = random.randint(0, self.loss)

            self.addLink(switches[l[0]], switches[l[1]], port1=port1, port2=port2,
                         bw=bw, delay=_d, loss=_l)

            links_info.setdefault(l, {"port1": port1, "port2": port2, "bw": bw, "delay": _d, "loss": _l})

        create_topo_links_info_xml(links_info_xml_path, links_info)

        # 添加host
        for i in self.node_idx:
            _h = self.addHost(f'h{i}', ip=f'10.0.0.{i}', mac=f'00.00.00.00.00.0{i}')
            self.addLink(_h, switches[i], port1=0, port2=self.host_port,
                         bw=self.bw4)

        # add snooper
        # snooper = self.addSwitch("s30")
        # for i in self.node_idx:
        #     self.addLink(snooper, switches[i], port1=i, port2=self.snooper_port)


class Nodes14Topo(Topo):
    def __init__(self, graph):
        super(Nodes14Topo, self).__init__()
        self.node_idx = graph.nodes
        self.edges_pairs = graph.edges

        self.random_bw = 30  # Gbps -> M * 10
        self.bw4 = 50  # host -- switch

        self.delay = 20  # ms
        self.loss = 10  # %

        self.host_port = 9
        self.snooper_port = 10

        # 添加交换机
        switches = {}
        for s in self.node_idx:
            switches.setdefault(s, self.addSwitch('s{0}'.format(s)))
            print('添加交换机:', s)

        switch_port_dict = generate_switch_port(graph)
        links_info = {}
        # 添加链路
        for l in self.edges_pairs:
            port1 = switch_port_dict[l[0]].pop(0) + 1
            port2 = switch_port_dict[l[1]].pop(0) + 1

            _bw = random.randint(5, self.random_bw)
            _d = str(random.randint(1, self.delay)) + 'ms'
            _l = random.randint(0, self.loss)

            self.addLink(switches[l[0]], switches[l[1]], port1=port1, port2=port2,
                         bw=_bw, delay=_d, loss=_l)

            links_info.setdefault(l, {"port1": port1, "port2": port2, "bw": _bw, "delay": _d, "loss": _l})

        create_topo_links_info_xml(links_info_xml_path, links_info)

        # 添加host
        for i in self.node_idx:
            _h = self.addHost(f'h{i}', ip=f'10.0.0.{i}', mac=f'00.00.00.00.00.0{i}')
            self.addLink(_h, switches[i], port1=0, port2=self.host_port,
                         bw=self.bw4)


def main(graph, topo, finish_file):
    print('===Remove old finish file')
    remove_finish_file(finish_file)

    net = Mininet(topo=topo, link=TCLink, controller=RemoteController, waitConnected=True, build=False)
    c0 = net.addController('c0', ip='127.0.0.1', port=6633)

    net.build()
    net.start()

    print("get hosts device list")
    hosts = get_mininet_device(net, graph.nodes, device='h')

    print("===Dumping host connections")
    dumpNodeConnections(net.hosts)
    print('===Wait ryu init')
    time.sleep(40)
    # 添加网关ip
    # run_ip_add_default(hosts)

    # net.pingAll()
    net_h1_ping_others(net)
    write_pinall_time(finish_file)

    # iperf脚本
    # hosts[1].cmd('iperf -s -u -p 1002 -1 &')
    # hosts[2].cmd('iperf -c 10.0.0.1 -u -p 1002 -b 20000k -t 30 &')
    print('===Run iperf scripts')
    t = threading.Thread(target=all_host_run_iperf, args=(hosts, iperf_path, finish_file), name='iperf', daemon=True)
    print('===Thread iperf start')
    t.start()
    # all_host_run_iperf(hosts, iperf_path)

    CLI(net)
    net.stop()


if __name__ == '__main__':
    xml_topology_path = r'./topologies/topology2.xml'
    links_info_xml_path = r'./links_info/links_info.xml'
    iperf_path = "./iperfTM"
    iperf_interval = 0
    finish_file = './finish_time.json'

    graph, nodes_num, edges_num = parse_xml_topology(xml_topology_path)
    # topo = GEANT23nodesTopo(graph)
    topo = Nodes14Topo(graph)

    setLogLevel('info')
    main(graph, topo, finish_file)

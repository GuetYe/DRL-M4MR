# -*- coding: utf-8 -*-
# @File    : generate_matrices.py
# @Date    : 2021-12-27
# @Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来-
# @From    :
import argparse
from pathlib import Path

import numpy as np
import numpy.random
from tmgen.models import modulated_gravity_tm
import matplotlib.pyplot as plt


def set_seed():
    numpy.random.seed(args.seed)


def generate_tm():
    tm = modulated_gravity_tm(args.num_nodes, args.num_tms, args.mean_traffic, args.pm_ratio, args.t_ratio,
                              args.diurnal_freq, args.spatial_variance, args.temporal_variance)

    mean_time_tm = []
    for t in range(args.num_tms):
        mean_time_tm.append(tm.at_time(t).mean())
        print(f"time: {t} h, mean traffic: {mean_time_tm[-1]}")

    # 构造一个(num_nodes, num_nodes, num_tms)的0-1均匀分布生成的矩阵
    _size = (args.num_nodes,) * 2
    _size += (args.num_tms,)

    temp = np.random.random(_size)
    mask = temp < args.communicate_ratio
    communicate_tm = tm.matrix * mask

    mean_communicate_tm = []
    for t in range(args.num_tms):
        mean_communicate_tm.append(communicate_tm[:, :, t].mean())
        print(f"time: {t} h, mean communicate nodes traffic: {mean_communicate_tm[-1]}")

    np_save(tm.matrix, "traffic_matrix")
    np_save(mean_time_tm, "mean_time_tm")

    np_save(communicate_tm, "communicate_tm")
    np_save(mean_communicate_tm, "mean_communicate_tm")

    plot_tm_mean(mean_time_tm, title="mean_time_tm")
    plot_tm_mean(mean_communicate_tm, title="mean_communicate_tm")


def np_save(file_data, file_name):
    Path('./tm_statistic').mkdir(exist_ok=True)
    np.save(f'./tm_statistic/{file_name}.npy', file_data)
    print(f'save {file_name}')


def plot_tm_mean(mean_list, x_label='time', y_label='mean_traffic', title='mean'):
    fig = plt.figure()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.title(title)
    x = list(range(len(mean_list)))
    y = mean_list
    plt.bar(x, y)
    Path("./figure").mkdir(exist_ok=True)
    plt.savefig(f"./figure/{title}.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate traffic matrices")
    parser.add_argument("--seed", default=2020, help="random seed")
    parser.add_argument("--num_nodes", default=14, help="number of nodes of network")
    parser.add_argument("--num_tms", default=24, help="total number of matrices")
    # 1.55 * 1e3 * 0.75
    parser.add_argument("--mean_traffic", default=5 * 10 ** 3 * 0.75, help="mean volume of traffic (Kbps)")
    parser.add_argument("--pm_ratio", default=1.5, help="peak-to-mean ratio")
    parser.add_argument("--t_ratio", default=0.75, help="trough-to-mean ratio")
    parser.add_argument("--diurnal_freq", default=1 / 24, help="Frequency of modulation")
    parser.add_argument("--spatial_variance", default=500,
                        help="Variance on the volume of traffic between origin-destination pairs")
    parser.add_argument("--temporal_variance", default=0.03, help="Variance on the volume in time")
    parser.add_argument("--communicate_ratio", default=0.7, help="percentage of nodes to communicate")
    args = parser.parse_args()

    # set_seed()
    # generate_tm()

    mean_time_tm = np.load(r"D:\WorkSpace\Hello_Myself\Hello_Multicast\RLMulticastProject\mininet\tm_statistic\tm_statistic\mean_time_tm.npy")
    plot_tm_mean(mean_time_tm)
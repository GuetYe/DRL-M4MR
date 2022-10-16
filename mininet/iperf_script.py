# -*- coding: utf-8 -*-
# @File    : iperf_script.py
# @Date    : 2022-02-10
# @Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来-
# @From    :
import argparse
import shutil
from pathlib import Path
import numpy as np


def read_npy(file=None):
    if file is None:
        file = args.file
    tms = np.load(file)
    return tms


def create_script(tms):
    label = 0
    tms = np.transpose(tms, (2, 0, 1))
    for tm in tms:
        # FOR CREATING FOLDERS PER TRAFFIC MATRIX
        Path(f'./iperfTM/TM-{label}').mkdir(parents=True, exist_ok=True)
        nameTM = Path(f'./iperfTM/TM-{label}')
        label += 1
        print('------', nameTM)
        Path.mkdir(nameTM, exist_ok=True)

        # --------------------FLOWS--------------------------
        # FOR CREATING FOLDERS PER NODE
        for i in range(len(tm[0])):
            Path.mkdir(nameTM / Path('Clients'), exist_ok=True)
            Path.mkdir(nameTM / Path('Servers'), exist_ok=True)

        # Default parameters
        time_duration = args.time_duration
        port = args.port
        ip_dest = args.ip_dest
        throughput = args.throughput  # take it in kbps from TM

        # UDP with time = 10s
        #   -c: ip_destination
        #   -b: throughput in k,m or g (Kbps, Mbps or Gbps)
        #   -t: time in seconds

        # SERVER SIDE
        # iperf3 -s

        # CLIENT SIDE with iperf3
        # iperf3 -c <ip_dest> -u -p <port> -b <throughput> -t <duration> -V -J

        # As we do not consider throughput in the same node, when src=dest the thro = 0
        for src in range(len(tm[0])):
            for dst in range(len(tm[0])):
                if src == dst:
                    print("src: ", src, "dst: ", dst)
                    tm[src][dst] = 0.0

        for src in range(1, len(tm[0]) + 1):
            with open(str(nameTM) + "/Clients/client_{0}.sh".format(str(src)), 'w') as fileClient:
                outputstring_a1 = "#!/bin/bash \necho Generating traffic..."
                fileClient.write(outputstring_a1)
                for dst in range(1, len(tm[0]) + 1):
                    throughput = float(tm[src - 1][dst - 1])
                    # throughput_g = throughput / (100) # scale the throughput value to mininet link capacities
                    temp1 = ''
                    if src != dst:
                        temp1 = ''
                        temp1 += '\n'
                        temp1 += 'iperf3 -c '
                        temp1 += '10.0.0.{0} '.format(str(dst))
                        if dst > 9:
                            temp1 += '-p {0}0{1} '.format(str(src), str(dst))
                        else:
                            temp1 += '-p {0}00{1} '.format(str(src), str(dst))
                        temp1 += '-u -b ' + str(format(throughput, '.2f')) + 'k'
                        # temp1 += ' -w 256k -t ' + str(time_duration)
                        temp1 += ' -t ' + str(time_duration)
                        temp1 += ' >/dev/null 2>&1 &\n'  # & at the end of the line it's for running the process in bkg
                        temp1 += 'sleep 0.4'
                    fileClient.write(temp1)

        # print(na)
        for dst in range(len(tm[0])):
            dst_ = dst + 1
            with open(str(nameTM) + "/Servers/server_{0}.sh".format(str(dst_)), 'w') as fileServer:
                outputstring_a2 = '#!/bin/bash \necho Initializing server listening...'
                fileServer.write(outputstring_a2)
                for src in range(len(tm[0])):
                    src_ = src + 1
                    temp2 = ''
                    if src != dst:
                        # n = n+1
                        temp2 = ''
                        temp2 += '\n'
                        temp2 += 'iperf3 -s '
                        if dst_ > 9:
                            temp2 += '-p {0}0{1} '.format(str(src_), str(dst_))
                        else:
                            temp2 += '-p {0}00{1} '.format(str(src_), str(dst_))
                        temp2 += '-1'
                        temp2 += ' >/dev/null 2>&1 &\n'  # & at the end of the line it's for running the process in bkg
                        temp2 += 'sleep 0.3'
                    fileServer.write(temp2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate traffic matrices")
    parser.add_argument("--seed", default=2020, help="random seed")
    # time_duration = 30
    #     port = 2022
    #     ip_dest = "10.0.0.1"
    #     throughput = 0.0  # take it in kbps from TM
    parser.add_argument("--time_duration", default=30, help="time_duration")
    parser.add_argument("--port", default=2022, help="port")
    parser.add_argument("--ip_dest", default="10.0.0.1", help="ip_dest")
    parser.add_argument("--throughput", default=0.0, help="take it in kbps from TM")
    parser.add_argument("--file",
                        default=r'tm_statistic/communicate_tm.npy',
                        help="take it in kbps from TM")
    args = parser.parse_args()
    shutil.rmtree("iperfTM")
    tms = read_npy()
    create_script(tms)

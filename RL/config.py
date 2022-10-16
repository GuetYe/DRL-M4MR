# -*- coding: utf-8 -*-
# @File    : config.py
# @Date    : 2022-05-18
# @Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来-
# @From    :
import time
from pathlib import Path
import platform
import torch
import numpy

sys_platform = platform.system()


class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TIME = time.strftime("%Y%m%d%H%M%S", time.localtime())

    # dtype
    NUMPY_TYPE = numpy.float32
    TORCH_TYPE = torch.float32

    # Net
    NUM_STATES = 14 ** 2 * 4
    NUM_ACTIONS = 14 ** 2

    PKL_STEP = 3
    PKL_NUM = 120
    PKL_START = 10  # index 从0开始
    PKL_CUT_NUM = PKL_START + PKL_NUM * PKL_STEP  # 结束index

    # DQN
    BATCH_SIZE = 8

    # memory pool
    MEMORY_CAPACITY = 1024 * 2

    # nsteps
    N_STEPS = 1

    # hyper-parameters
    LR = 1e-5  # learning rate
    REWARD_DECAY = 0.9  # gamma

    USE_DECAY = True
    E_GREEDY_START = 1  # epsilon
    E_GREEDY_FINAL = 0.01  # epsilon
    E_GREEDY_DECAY = 700  # epsilon
    E_GREEDY = [E_GREEDY_START, E_GREEDY_FINAL, E_GREEDY_DECAY]

    E_GREEDY_ORI = 0.2

    TAU = 1

    EPISODES = 4000  # 训练代数
    UPDATE_TARGET_FREQUENCY = 10

    REWARD_DEFAULT = [1.0, 0.1, -0.1, -1]

    DISCOUNT = 0.9

    BETA1 = 1
    BETA2 = 1
    BETA3 = 1

    A_STEP = 0
    B_STEP = None

    A_PATH = 0
    B_PATH = 1

    START_SYMBOL = 1
    END_SYMBOL = 2
    STEP_SYMBOL = 1
    BRANCH_SYMBOL = 2

    # control
    # START_LEARN = 200
    CLAMP = False  # 梯度裁剪

    # file path  and pkl path
    if sys_platform == "Windows":
        xml_topology_path = Path(
            r'D:\WorkSpace\Hello_Myself\Hello_Multicast\RLMulticastProject\mininet\topologies\topology2.xml')
        pkl_weight_path = Path(
            r"D:\WorkSpace\Hello_Myself\Hello_Multicast\RLMulticastProject\ryu\pickle\2022-03-11-19-40-21")
            

    else:
        xml_topology_path = Path(r'/home/dell/RLMulticastProject/mininet/topologies/topology2.xml')
        pkl_weight_path = Path(r"/home/dell/RLMulticastProject/ryu/pickle/2022-03-11-19-40-21")

    # nodes
    start_node = 12
    end_nodes = [2, 4, 11]

    @classmethod
    def set_num_states_actions(cls, state_space_num, action_space_num):
        cls.NUM_STATES = state_space_num
        cls.NUM_ACTIONS = action_space_num

    @classmethod
    def log_params(cls, logger):
        rewards_info = "\n===rewards===\n" + \
                       f" REWARD_DEFAULT:{cls.REWARD_DEFAULT}\n"
        lr_info = "===LR===\n" + \
                  f" LR:{cls.LR}\n"
        episodes_info = "===EPISODES===\n" + \
                        f" EPISODES:{cls.EPISODES}\n"
        batchsize_info = "===BATCH_SIZE===\n" + \
                         f" BATCH_SIZE:{cls.BATCH_SIZE}\n"

        update_infp = "===UPDATE_TARGET_FREQUENCY===\n" + \
                      f" UPDATE_TARGET_FREQUENCY:{cls.UPDATE_TARGET_FREQUENCY}\n"
        gamma_info = "===gamma==\n" + \
                     f" REWARD_DECAY:{cls.REWARD_DECAY}\n"
        nsteps_info = "===nsteps===\n" + \
                      f" N_STEPS:{cls.N_STEPS}\n"
        egreedy_info = "===egreedy===\n" + \
                       f" [E_GREEDY_START, E_GREEDY_FINAL, E_GREEDY_DECAY:{cls.E_GREEDY}\n"
        pickle_info = "===Pickle Param===\n" + \
                      f" PKL_START, PKL_NUM, PKL_STEP:{cls.PKL_STEP}, {cls.PKL_NUM}, {cls.PKL_START}\n"

        env_info = "===ENV===\n" + \
                   f" DISCOUNT:{cls.DISCOUNT},\n BETA1, BETA2, BETA3：{cls.BETA1},{cls.BETA2},{cls.BETA3}\n"
        logger.info(
            rewards_info + lr_info + episodes_info + batchsize_info + update_infp + gamma_info + nsteps_info + egreedy_info + pickle_info + env_info)

        # logger.info(cls.__dict__)

    @classmethod
    def set_lr(cls, lr):
        cls.LR = lr

    @classmethod
    def set_nsteps(cls, nsteps):
        cls.N_STEPS = nsteps

    @classmethod
    def set_batchsize(cls, batchsize):
        cls.BATCH_SIZE = batchsize

    @classmethod
    def set_egreedy(cls, egreedy):
        cls.E_GREEDY_DECAY = egreedy

    @classmethod
    def set_gamma(cls, gamma):
        cls.REWARD_DECAY = gamma

    @classmethod
    def set_update_frequency(cls, update_frequency):
        cls.UPDATE_TARGET_FREQUENCY = update_frequency

    @classmethod
    def set_tau(cls, tau):
        cls.TAU = tau

    @classmethod
    def set_rewards(cls, rewards):
        cls.REWARD_DEFAULT = rewards

    @classmethod
    def print_cls_dict(cls):
        print(cls.__dict__)

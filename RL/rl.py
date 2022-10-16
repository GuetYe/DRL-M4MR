# -*- coding: utf-8 -*-
# @File    : rl.py
# @Date    : 2022-05-18
# @Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来-
# @From    :
import math
import os
import pickle
import random

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from replymemory import Transition, ExperienceReplayMemory, PrioritizedReplayMemory


class DQN:
    """
     Deep Q Network
     调用 self.flat_state(state) 将二维状态展平成一维状态
     调用 self.choose_action(state) 从policy_net选择一个动作
     调用 self.learn(state, action, reward, next_state)学习
    """

    def __init__(self,
                 conf,
                 net):
        """
            :params conf: config.Config() 实例
            :params net: net.Net 类
        """

        self.state_num = conf.NUM_STATES
        self.action_num = conf.NUM_ACTIONS

        self.lr = conf.LR
        self.use_decay = conf.USE_DECAY
        self.gamma = conf.REWARD_DECAY
        self.epsilon_start, self.epsilon_final, self.epsilon_decay = conf.E_GREEDY
        self.epsilon = conf.E_GREEDY_ORI

        self.tau = conf.TAU

        self.device = conf.DEVICE
        self.batch_size = conf.BATCH_SIZE
        self.memory_capacity = conf.MEMORY_CAPACITY

        # self.start_learn = config.START_LEARN
        self.clamp = conf.CLAMP
        self.torch_type = conf.TORCH_TYPE

        self.experiment_time = conf.TIME

        self.n_steps = conf.N_STEPS
        self.n_step_buffer = []

        self.policy_net, self.target_net = self.initialize_net(net)

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[2500, ], gamma=0.1)  # [2500, 5000]
        # self.loss_func = nn.MSELoss()
        # HuberLoss 当delta=1时，等价于smoothl1一样
        self.loss_func = nn.SmoothL1Loss(reduction='none')
        self.memory_pool = self.initialize_replay_memory("PrioritizedReplayMemory")

        self.learning_step_count = 0
        self.update_target_count = 0
        self.update_target_frequency = conf.UPDATE_TARGET_FREQUENCY

        self.losses = []

        self.train_or_eval = 'train'
        self.change_model_mode('train')

        self.move_net_to_device()

        Path("./saved_agents").mkdir(exist_ok=True)

    def initialize_replay_memory(self, mode):
        """
            初始化 回放池
        :param mode: ExperienceReplayMemory 和 PrioritizedReplayMemory
        :return: 经验回放的类的实例
        """
        if mode == 'ExperienceReplayMemory':
            return ExperienceReplayMemory(self.memory_capacity, self.torch_type)
        elif mode == 'PrioritizedReplayMemory':
            return PrioritizedReplayMemory(self.torch_type, self.memory_capacity)

    def initialize_net(self, net):
        """
            初始化policy网络 和 target网络
        :param net: net.py中的net: nn.Module
        :return: policy_net, target_net: net的实例
        """
        policy_net = net(self.state_num, self.action_num)
        target_net = net(self.state_num, self.action_num)
        return policy_net, target_net

    def move_net_to_device(self):
        """
            将网络移动到设备
        """
        self.policy_net.to(self.device)
        self.target_net.to(self.device)

    def change_model_mode(self, train_or_eval='train'):
        """
            改变模型的模式，model.train() 或者 model.eval()
        :param train_or_eval: ['train', 'eval']
        :return: None
        """
        assert train_or_eval in ['train', 'eval']
        if train_or_eval == 'train':
            self.policy_net.train()
            self.target_net.train()

        elif train_or_eval == 'eval':
            self.policy_net.eval()
            self.target_net.eval()

        self.train_or_eval = train_or_eval

    def decay_epsilon(self, epoch, mode='exp'):
        """
            decay e-greedy
        :param epoch: 代数
        :param mode: 模式
        :return: eps
        """
        if mode == 'exp':
            eps = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * math.exp(
                -1. * epoch / self.epsilon_decay)
        else:
            eps = min(abs(self.epsilon_start - self.epsilon_final * epoch), self.epsilon_final)
        return eps

    def choose_action(self, s, epoch):
        """
            选择动作 e_greedy
            2022/3/20  action_value * torch.from_numpy(mask) 如果全为0 那么选的时候可能会选不应该选的
        :param s: 状态
        :param epoch: 代数
        :return: action
        """
        if self.use_decay:
            epsilon = self.decay_epsilon(epoch)
        else:
            epsilon = self.epsilon
        # temp = random.random()
        temp = np.random.rand()
        if temp >= epsilon:  # greedy policy
            action_value = self.policy_net.forward(s.to(self.device))
            # shape [1, actions_num]; max -> (values, indices)
            act_node = torch.max(action_value, 1)[1]  # 最大值的索引
        else:  # random policy
            act_node = torch.from_numpy(np.random.choice(np.array(range(self.action_num)), size=1)).to(self.device)
        # array([indice], dtype)
        act_node = act_node[0]
        return act_node

    def choose_max_action(self, s):
        """
            test时用，每次选择最大action-state value的动作
        :param s: 状态
        :return act_node: action index
        """
        action_value = self.policy_net.forward(s.to(self.device))
        # shape [1, actions_num]; max -> (values, indices)
        act_node = torch.max(action_value, 1)[1]  # 最大值的索引
        return act_node

    def store_transition_in_memory(self, s, a, r, s_):
        """
            存放到经验池中
        :param s: 状态s
        :param a: 动作a
        :param r: 回报r
        :param s_: 状态s撇
        :return: None
        """
        self.n_step_buffer.append((s, a, r, s_))
        if len(self.n_step_buffer) < self.n_steps and s_ is not None:
            return
        R = sum([self.n_step_buffer[i][2] * (self.gamma ** i) for i in range(self.n_steps)])
        s, a, _, _ = self.n_step_buffer.pop(0)

        self.memory_pool.push(s, a, R, s_)

    def finish_n_steps(self):
        """
            将最后小于 n steps 的那几步，存入memory
        """
        while len(self.n_step_buffer) > 0:
            R = sum([self.n_step_buffer[i][2] * (self.gamma ** i) for i in range(len(self.n_step_buffer))])
            s, a, _, _ = self.n_step_buffer.pop(0)
            self.memory_pool.push(s, a, R, None)

    def get_batch_vars(self):
        """
            获得batch的state，action，reward，next_state
        :return: state_batch, action_batch, reward_batch, non_final_next_states, non_final_mask, indices, weight
        """
        transitions, indices, weights = self.memory_pool.sample(self.batch_size)

        batch = Transition(*zip(*transitions))
        # 竖着放一起 (B, Hin)
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

        # 计算非最终状态的掩码并连接批处理元素(最终状态将是模拟结束后的状态）
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)

        try:
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(self.device)
            non_flag = False
        except Exception as e:
            non_final_next_states = None
            non_flag = True

        return state_batch, action_batch, reward_batch, non_final_next_states, non_flag, non_final_mask, indices, weights

    def calculate_loss(self, batch_vars):
        """
            计算损失
        :param batch_vars: state_batch, action_batch, reward_batch, non_final_next_states, non_final_mask, indices, weight
        :return: loss
        """
        batch_state, batch_action, batch_reward, non_final_next_states, non_flag, non_final_mask, indices, weights = batch_vars

        # 从policy net中根据状态s，获得执行action的values
        _out = self.policy_net(batch_state)
        state_action_values = torch.gather(_out, 1, batch_action.type(torch.int64))

        with torch.no_grad():
            # 如果non_final_next_states是全None，那么max_next_state_values就全是0
            max_next_state_values = torch.zeros(self.batch_size, dtype=self.torch_type, device=self.device).unsqueeze(1)
            if not non_flag:
                # 从target net中根据非最终状态，获得相应的value值
                max_next_action = self.target_net(non_final_next_states).max(dim=1)[1].view(-1, 1)
                max_next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1,
                                                                                                      max_next_action)
            # 计算期望的Q值
            expected_state_action_values = (max_next_state_values * self.gamma) + batch_reward

        self.memory_pool.update_priorities(indices, (
                state_action_values - expected_state_action_values).detach().squeeze(1).abs().cpu().numpy().tolist())

        # 计算Huber损失
        loss = self.loss_func(state_action_values, expected_state_action_values) * weights.unsqueeze(1)
        loss = loss.mean()

        return loss

    def learn(self, s, a, r, s_, episode):
        """
            从transition中学习
        :param s: 状态s
        :param a: 动作a
        :param r: 回报r
        :param s_: 状态s撇
        :return: None
        """
        if self.train_or_eval == 'eval':
            return

        self.store_transition_in_memory(s, a, r, s_)

        if len(self.memory_pool) < self.batch_size:
            return

        batch_vars = self.get_batch_vars()
        loss = self.calculate_loss(batch_vars)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        if self.clamp:
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        self.update_target_model(episode)
        self.append_loss_item(loss.data)

    def update_target_model(self, episode):
        """
            更新target网络
        :return: None
        """
        self.update_target_count += 1
        self.update_count = self.update_target_count % self.update_target_frequency
        # self.update_count = episode % self.update_target_frequency
        if self.update_count == 0:
            # self.target_net.load_state_dict(self.policy_net.state_dict())
            self._soft_update()

    def _soft_update(self):
        """
            软更新
        """
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def append_loss_item(self, loss):
        """
            将loss添加到 losses中
        :param loss: 损失
        :return: None
        """

        self.losses.append(loss)

    def save_weight(self, name):
        """
            保存模型 和 优化器的参数
        :return: None
        """

        torch.save(self.policy_net.state_dict(),
                   f'./saved_agents/policy_net-{name}.pt')
        torch.save(self.target_net.state_dict(),
                   f'./saved_agents/target_net-{name}.pt')
        torch.save(self.optimizer.state_dict(),
                   f'./saved_agents/optim-{name}.pt')

    def load_weight(self, fname_model, fname_optim):
        """
            加载模型和优化器的参数，policy_net网络参数，target_net网络参数，optimizer参数
        :return: None
        """
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if os.path.isfile(fname_model):
            self.policy_net.load_state_dict(torch.load(fname_model, map_location=device))
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            raise FileNotFoundError

        if fname_optim is not None and os.path.isfile(fname_optim):
            self.optimizer.load_state_dict(torch.load(fname_optim))

    def save_replay(self):
        """
            保存经验池的数据， pickle形式
        :return: None
        """
        pickle.dump(self.memory_pool, open(f'./saved_agents/exp_replay_agent-{self.experiment_time}.pkl', 'wb'))

    def load_replay(self, fname):
        """
            加载经验池的数据
        :return: Nones
        """
        if os.path.isfile(fname):
            self.memory_pool = pickle.load(open(fname, 'rb'))

# -*- coding: utf-8 -*-
# @File    : replymemory.py
# @Date    : 2022-05-18
# @Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来-
# @From    :
import random
import operator
from collections import namedtuple
import torch
import numpy as np

from config import Config

# 使用具名元组 快速建立一个类
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))


class ExperienceReplayMemory:
    def __init__(self, capacity, torch_type) -> None:
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.torch_type = torch_type

    def push(self, *args):
        """保存变换"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        s, a, r, s_ = args
        self.memory[self.position] = Transition(s,
                                                a.reshape(1, -1),
                                                torch.tensor(r, dtype=self.torch_type).reshape(1, -1),
                                                s_)
        # self.memory[self.position] = Transition(*torch.tensor(args))
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size), None, None

    def clean_memory(self):
        self.memory = []

    def __len__(self):
        return len(self.memory)


class SegmentTree:
    def __init__(self, capacity, operation, neutral_element):
        assert capacity > 0 and capacity & (capacity - 1) == 0
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, query_start, query_end, node, node_start, node_end):
        if query_start == node_start and query_end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if query_end <= mid:
            return self._reduce_helper(query_start, query_end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= query_start:
                return self._reduce_helper(query_start, query_end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(query_start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, query_end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        if end is None:
            end = self._capacity
        if end <= 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        try:
            assert 0 <= prefixsum <= self.sum() + np.finfo(np.float32).eps
        except AssertionError:
            print(f"Prefix sum error: {prefixsum}")
            exit()
        idx = 1
        while idx < self._capacity:
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        return super(MinSegmentTree, self).reduce(start, end)


class PrioritizedReplayMemory:
    def __init__(self, torch_type, size, alpha=0.6, beta_start=0.4, beta_frames=70000*Config.PKL_NUM):
        self.torch_type = torch_type
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

        assert alpha >= 0
        self._alpha = alpha

        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def __len__(self):
        return len(self._storage)

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, *data):
        s, a, r, s_ = data
        data = Transition(s,
                          a.reshape(1, -1),
                          torch.tensor(r, dtype=self.torch_type).reshape(1, -1),
                          s_)

        idx = self._next_idx
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _encode_sample(self, idxes):
        return [self._storage[i] for i in idxes]

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size):
        idxes = self._sample_proportional(batch_size)
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()

        beta = self.beta_by_frame(self.frame)

        self.frame += 1
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = torch.tensor(weights, device=self.device, dtype=self.torch_type)
        encoded_sample = self._encode_sample(idxes)
        return encoded_sample, idxes, weights

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = (priority + np.finfo(np.float32).eps) ** self._alpha
            self._it_min[idx] = (priority + np.finfo(np.float32).eps) ** self._alpha
            self._max_priority = max(self._max_priority, (priority + np.finfo(np.float32).eps))

# # -*- coding: utf-8 -*-
# # @File    : env_config.py
# # @Date    : 2022-05-18
# # @Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来-
# # @From    :
# # done_reward, step_reward, alley_reward, hell_reward
# # [1.0, 0.01, -0.001, -1], [1.0, 0.1, -0.001, -1], [1.0, 1.0, -0.001, -1]
#
# reward_default = [1.0, 0.1, -0.1, -1]
#
# DISCOUNT = 0.9
#
# BETA1 = 1
# BETA2 = 1
# BETA3 = 1
#
# A_STEP = 0
# B_STEP = None
#
# A_PATH = 0
# B_PATH = 1
#
# START_SYMBOL = 1
# END_SYMBOL = 2
# STEP_SYMBOL = 1
# BRANCH_SYMBOL = 2
#
#
# def traverse_reward_list(reward_list):
#     for idx in range(len(reward_list)):
#         yield reward_list[idx]
#
# def set_rewards(rewards):
#     reward_default = rewards
#     return

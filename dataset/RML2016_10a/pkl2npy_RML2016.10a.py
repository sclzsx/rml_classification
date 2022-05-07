# -*- coding: UTF8 -*-

# ------------------------------------------------------------------------
# File Name:        pkl2npy_RML2016.10a.py
# Author:           JunJie Ren
# Version:          v1.0
# Created:          2021/1/21
# Description:      — — — — — — — — — — — — — — — — — — — — — — — — — — — 
#                           --> 小样本信号(RML2016.10a)系列代码 <--     
#                   -- 对RML2016.10a数据集的源文件.pkl进行处理：对同一信噪比
#                   不同调制下的数据，随机采样平分为测试集(大小11*500)、训练集
#                   (大小11*500)；再对训练集随机采样形成每种调制下含10,20,...,
#                   100个样本的小样本训练集；分别将测试集、小样本训练集按不同信
#                   噪比分文件夹保存为.npy
#                   -- 保存格式：{"调制方式1"：DATA, "调制方式2": DATA, ...}、
#                   例如：SNR=8dB，每类100个：{b'BPSK':(100, 2, 128), ...}
#                   -- 数据集描述：11种调制信号，每种调制包含20种信噪比，每种信
#                   噪比有1000个样本，每个样本有I和Q两路信号，每路信号包含128个
#                   点，所以数据集大小为：220000×2×128，共220000条数据。
#                   — — — — — — — — — — — — — — — — — — — — — — — — — — — 
# Module called:    None
# Function List:    None
# Class List:       None
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# History:
#      |  <author>  | <version> |   <time>   |          <desc>
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#  <0> | JunJie Ren |   v1.0    | 2021/01/21 |   Achieve all functions
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#  <1> | JunJie Ren |   v1.1    | 2021/04/23 |     Half-data training
# ------------------------------------------------------------------------

import pickle
import numpy as np
import random
import os
import sys
path = 'RML2016.10a_dict.pkl'

f = open(path, 'rb')
f_data = pickle.load(f, encoding='bytes')
# print("Data length：", len(f_data))
# for i in f_data.keys():
#     print(i, ':', len(f_data[i]))
# sys.exit()

data_train = {}
data_test = {}


train_idx = random.sample(range(1000), 500)
test_idx = []
for i in range(1000):
    if i not in train_idx:
        test_idx.append(i)

for name, samples in f_data.items():
    MOD, SNR = name[0], name[1]  # name: (b'QPSK', -8) # (调制方式，信噪比)
    if SNR not in data_train:
        data_train[SNR] = {}
        data_test[SNR] = {}
    data_train[SNR][MOD] = samples[train_idx, :, :]
    data_test[SNR][MOD] = samples[test_idx, :, :]

print("\ntrain:")
for snr in data_train.keys():
    print(snr, ':', len(data_train[snr]), data_train[snr][b'BPSK'].shape)

print("\ntest:")
for snr in data_test.keys():
    print(snr, ':', len(data_test[snr]), data_test[snr][b'BPSK'].shape)
# sys.exit()


# 保存测试数据至.npy
for SNR, mod_dict in data_test.items():
    if not os.path.exists("{}dB_SNR/".format(SNR)):
        os.makedirs("{}dB_SNR/".format(SNR))
    test_path = "{}dB_SNR/{}dB-SNR_500-test.npy".format(SNR, SNR)
    np.save(test_path, mod_dict)

    print("------------------> Have saved test file:" + test_path)
print("\n\nFinished all test files!!!\n\n")

for SNR, mod_dict in data_train.items():
    if not os.path.exists("{}dB_SNR/".format(SNR)):
        os.makedirs("{}dB_SNR/".format(SNR))
    train_path = "{}dB_SNR/{}dB-SNR_500-train.npy".format(SNR, SNR)
    np.save(train_path, mod_dict)

    print("------------------> Have saved train file:" + train_path)
print("\n\nFinished all train files!!!\n\n")

# # 保存小样本训练数据至.npy
# # for i in range(1, 11):
# # 不同样本数量下的SNR数据分别保存
# # samplesNum = i * 10
# samplesNum = 100
# few_shot_lable = random.sample(range(500), samplesNum)
#
# for SNR, mod_dict in data_train.items():
#     # 不同SNR下的调制数据分别保存
#     few_shot_data = {}  # 每类samplesNum个、信噪比为SNR下的小样本训练数据
#     for MOD, samples in mod_dict.items():
#         # print(SNR, MOD, samples.shape)
#         few_shot_data[MOD] = samples[few_shot_lable, :, :]
#
#     # 保存小样本训练数据至.npy
#     train_path = "{}dB_SNR/{}dB-SNR_{}-samples.npy".format(SNR, SNR, samplesNum)
#     np.save(train_path, few_shot_data)
#
#     print("------------------> Have saved train file:" + train_path)
# print("Have saved {} samplesNum!\n".format(samplesNum))
# print("\n\nFinished all train files!!!\n\n")

# -*- coding: UTF-8 -*-

import numpy as np
from numpy import linalg as la

# np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

raw_column_size = 40
num_class = 5
sub_sample = 5000


def data_prepare():
    cols = range(1, 121)
    dataset = np.loadtxt('/media/whsse/软件/Paper/罗丹师姐/TotalData_25000_with_5_label.txt', dtype=float, usecols=cols)
    print(dataset.shape)

    new_dataset = np.empty((25000, raw_column_size))
    for i in range(raw_column_size):
        new_dataset[:, i] = la.norm(dataset[:, 3 * i: 3 * i + 3], axis=1)

    np_data = np.empty((num_class, sub_sample, raw_column_size))   # 原始数据 (25, 100, 40)
    np_label = np.ones((sub_sample, 1))               # 标签  (500, 1)
    np_sample = np.empty((num_class, sub_sample, raw_column_size + 1))

    for i in range(0, num_class):
        np_data[i] = new_dataset[i * sub_sample: i * sub_sample + sub_sample, :]
        np_sample[i] = np.concatenate((np_data[i], i * np_label), axis=1)  # 合并为样本

    print("=== Data Prepare ===")
    total_data = np_sample.reshape(num_class * sub_sample, raw_column_size + 1)  # (2500, 41)

    return total_data


if __name__ == '__main__':

    # ld_data = np.load('new_all_wifi_for_raw.npy')
    # infs = [6742, 7324, 15447, 24700]
    # deal_data = np.delete(ld_data, infs, axis=0)
    # print(deal_data.shape)
    # print(np.argwhere(np.isinf(deal_data)))
    # print(np.isfinite(deal_data).all())
    # np.save('new_clear_data_for_raw.npy', deal_data)

    data = np.load('new_clear_data_for_fea81.npy')
    print(data)
    print(data.shape)
    print(np.argwhere(np.isinf(data)))

    # fea = np.load('new_all_wifi_for_fea81.npy')
    # nor = np.load('new_all_wifi_for_raw.npy')
    # deal_data = np.concatenate((nor[:, 0:40], fea), axis=1)  # 合并为样本
    # print(deal_data)
    # np.save('new_all_wifi_for_fea81+norm.npy', deal_data)

    print("Done...")

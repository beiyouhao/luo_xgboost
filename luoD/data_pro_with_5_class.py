# -*- coding: UTF-8 -*-

import numpy as np

# np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

raw_column_size = 120
num_class = 5
sub_sample = 1000
fea_size = 27


def data_prepare():
    cols = range(1, 121)
    dataset = np.loadtxt('/media/whsse/软件/Paper/罗丹师姐/Wifi1537152560239_pos2.txt', dtype=float, usecols=cols)
    print(dataset.shape)

    np_data = np.empty((num_class, sub_sample, raw_column_size))   # 原始数据 (5, 500, 120)
    np_label = np.ones((sub_sample, 1))                            # 标签  (500, 1)
    np_sample = np.empty((num_class, sub_sample, raw_column_size + 1))

    for i in range(0, num_class):
        np_data[i] = dataset[i * 1000: i * 1000 + 1000, :]
        np_sample[i] = np.concatenate((np_data[i], i * np_label), axis=1)  # 合并为样本

    print("=== Data Prepare ===")
    total_data = np_sample.reshape(num_class * sub_sample, raw_column_size + 1)  # (2500, 121)
    return total_data


if __name__ == '__main__':

    res = data_prepare()
    print(res)
    np.save('new_wifi_pos2.npy', res)

    # datax = np.load('ld_fea120.npy')
    print("Done...")

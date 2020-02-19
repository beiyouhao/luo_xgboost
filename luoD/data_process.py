# -*- coding: UTF-8 -*-

import numpy as np

# np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

raw_column_size = 120
num_class = 25
sub_sample = 1000


# 最大最小值标准化
def normalize_cols(sample):
    col_max = sample.max(axis=0)  # max value in each column
    col_min = sample.min(axis=0)  # min value in each column
    return (sample - col_min) / (col_max - col_min)


def data_prepare():
    cols = range(1, 121)
    dataset = np.loadtxt('D:\Paper\罗丹师姐\TotalData_25000_with_25_label.txt', dtype=float, usecols=cols)
    print(dataset.shape)

    print("inf1:", np.isinf(dataset).any())

    # print(type(pos))
    # tmp = pos[0, 0]
    # listx = [tmp]
    # for i in range(len(pos)):
    #     if tmp != pos[i, 0]:
    #         tmp = pos[i, 0]
    #         listx.append(tmp)
    # print(listx)
    # print(listx[0])

    np_data = np.empty((num_class, sub_sample, raw_column_size))   # 原始数据 (5, 500, 120)
    np_label = np.ones((sub_sample, 1))                            # 标签  (500, 1)
    np_sample = np.empty((num_class, sub_sample, raw_column_size + 1))

    print("=== Data Prepare ===")
    for i in range(0, num_class):
        np_data[i] = dataset[i * 1000: i * 1000 + 1000, :]
        # 数据清洗
        if np.isinf(np_data[i]).any():
            inf_pos = np.argwhere(np.isinf(np_data[i]))
            l1 = inf_pos[:, 0].tolist()
            inf_index = sorted(set(l1), key=l1.index)
            np_data[i, inf_index, :] = np.mean(np_data[i, 1:inf_index[0], :], axis=0)

        np_data[i] = normalize_cols(np_data[i])
        np_sample[i] = np.concatenate((np_data[i], i * np_label), axis=1)  # 合并为样本

    total_data = np_sample.reshape(num_class * sub_sample, raw_column_size + 1)  # (2500, 121)
    print("inf2:", np.isinf(total_data).any())
    return total_data


if __name__ == '__main__':

    res = data_prepare()
    print(res.shape)
    print(np.argwhere(np.isinf(res)))
    print(np.argwhere(np.isnan(res)))
    print(res)
    np.save('whsse_normalized_for_raw_25.npy', res)

    # datax = np.load('new_normalized_for_raw_123.npy')
    # print(datax)

    print("Done...")

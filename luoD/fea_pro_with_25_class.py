# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd

# np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

raw_column_size = 120
fea_size = 27
num_class = 5
sub_sample = 5000


def getIntergral(data, freq):
	s = 0
	for i in np.arange(1, len(data)):
		s += (data[i - 1] + data[i]) / 2 * freq
	return s


def getDoubleIntegral(data, freq):
	a = [0]
	s = 0
	for i in np.arange(1, len(data)):
		s += (data[i - 1] + data[i]) / 2 * freq
		a.append(s)
	r = getIntergral(a, freq)
	return r


def getMeanCrossingRate(data):
	mean = np.mean(data)
	r = 0
	for i in np.arange(0, len(data) - 1):
		if data[i] > mean and data[i + 1] < mean:
			r += 1
		if data[i] < mean and data[i + 1] > mean:
			r += 1
	return r


def getFrequencyFeature(data):
	fft = np.fft.fft(data, 100)
	fft = np.abs(fft)
	feature = fft[:7].tolist()
	return feature


# 共计27个特征
def get_feature(sample):
    feature = [sample.mean(), sample.std(ddof=0), sample.var(),
               sample.median(), sample.min(), sample.max(), sample.ptp(),                   # 中位数 最小值 最大值 两者之差
               sample.quantile(0.25), sample.quantile(0.75),                                # 四分差
               sample.kurt(), sample.skew(), sample.autocorr(), (sample.diff() > 0).sum(),  # 峰度 偏度 自相关系数
               sample.sem(), sample.mad(), sample.sum(),
               np.sqrt((np.sum(sample ** 2) / len(sample))),            # RMS
               getIntergral(sample, 0.01), getDoubleIntegral(sample, 0.01), getMeanCrossingRate(sample)]  # 积分
    for ele in getFrequencyFeature(sample):
        feature.append(ele)
    return feature


def feature_extraction(x):
    df_data = pd.DataFrame(x)
    fea = np.empty((len(x), 27))  # 特征数
    for index, row in df_data.iterrows():
        fea[index] = get_feature(row)
    return fea


def data_prepare():
    cols = range(1, 121)
    dataset = np.loadtxt('/media/whsse/软件/Paper/罗丹师姐/TotalData_25000_with_5_label.txt', dtype=float, usecols=cols)
    print(dataset.shape)

    # np_data = np.empty((num_class, sub_sample, raw_column_size / 3))   # 原始数据

    fea_x = np.empty((num_class, sub_sample, fea_size))  # 特征数据
    fea_y = np.empty((num_class, sub_sample, fea_size))  # 特征数据
    fea_z = np.empty((num_class, sub_sample, fea_size))  # 特征数据
    np_label = np.ones((sub_sample, 1))                  # 标签  (100, 1)
    np_sample = np.empty((num_class, sub_sample, fea_size * 3 + 1))

    index_x = range(0, raw_column_size, 3)
    index_y = range(1, raw_column_size, 3)
    index_z = range(2, raw_column_size, 3)
    for i in range(0, num_class):
        data_x = dataset[i * 5000: i * 5000 + 5000, index_x]  # shape = (5000, 40)
        data_y = dataset[i * 5000: i * 5000 + 5000, index_y]  # shape = (5000, 40)
        data_z = dataset[i * 5000: i * 5000 + 5000, index_z]  # shape = (5000, 40)
        fea_x[i] = feature_extraction(data_x)   # shape = (100, 27)
        fea_y[i] = feature_extraction(data_y)   # shape = (100, 27)
        fea_z[i] = feature_extraction(data_z)   # shape = (100, 27)
        np_sample[i] = np.concatenate((fea_x[i], fea_y[i], fea_z[i], i * np_label), axis=1)  # 合并为样本

    print("=== Data Prepare ===")
    total_data = np_sample.reshape(num_class * sub_sample, fea_size * 3 + 1)  # (2500, 81+1)
    return total_data


if __name__ == '__main__':

    res = data_prepare()
    print(res.shape)
    np.save('new_all_wifi_for_fea81.npy', res)

    # datanorm = np.load('stepsize8_for_norm.npy')
    # x = datanorm[:, 0:160]
    # y = datanorm[:, 160]
    # datafeaxyz = np.load('stepsize8_for_fea_xyz.npy')
    # deal_data = np.concatenate((x, datafeaxyz), axis=1)  # 合并为样本

    # print(deal_data.shape)
    # print(deal_data)

    # data_norm = np.load('ld_raw40_for_norm.npy')
    # # mean_norm = np.mean(data_norm[:, 0:40], axis=1)
    #
    # deal_data = np.concatenate((data_norm[:, 0:40], dataxyz), axis=1)  # 合并为样本
    # print(deal_data)

    # np.save('stepsize8_fea&norm.npy', deal_data)

    print("Done...")

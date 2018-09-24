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

    np_data = np.empty((num_class, sub_sample, raw_column_size))   # 原始数据
    np_fea = np.empty((num_class, sub_sample, fea_size))           # 特征数据
    np_label = np.ones((sub_sample, 1))                            # 标签  (100, 1)
    np_sample = np.empty((num_class, sub_sample, fea_size + 1))

    for i in range(0, num_class):
        np_data[i] = dataset[i * 5000: i * 5000 + 5000, :]
        np_fea[i] = feature_extraction(np_data[i])
        np_sample[i] = np.concatenate((np_fea[i], i * np_label), axis=1)  # 合并为样本

    print("=== Data Prepare ===")
    total_data = np_sample.reshape(num_class * sub_sample, fea_size + 1)  # (25000, 28)
    return total_data


if __name__ == '__main__':
    res = data_prepare()
    print(res.shape)
    np.save('new_all_wifi_for_fea27.npy', res)
    # datax = np.load('ld_fea120.npy')
    print("Done...")

# -*- coding: UTF-8 -*-

import numpy as np
from numpy import linalg as la
import pandas as pd

# np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

raw_col = 120

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


if __name__ == '__main__':
    data = np.load('new_clear_data_for_raw.npy')

    x = data[:, 0:raw_col]
    y = data[:, raw_col]
    print(x.shape)
    y = y.reshape(24996, 1)
    print(y)

    index_x = range(0, raw_col, 3)
    index_y = range(1, raw_col, 3)
    index_z = range(2, raw_col, 3)
    # 取样本x的对应索引
    data_x = x[:, index_x]
    data_y = x[:, index_y]
    data_z = x[:, index_z]
    print(data_x.shape, data_y.shape, data_z.shape)
    fea_x = feature_extraction(data_x)  # shape = (100, 27)
    fea_y = feature_extraction(data_y)  # shape = (100, 27)
    fea_z = feature_extraction(data_z)  # shape = (100, 27)
    print(fea_x.shape, fea_y.shape, fea_z.shape)
    sample1 = np.concatenate((fea_x, fea_y, fea_z, y), axis=1)
    print(sample1)
    np.save('new_clear_data_for_fea81.npy', sample1)
    print("Done...")

    # new_dataset = np.empty((24996, 40))
    # for i in range(40):
    #     new_dataset[:, i] = la.norm(x[:, 3 * i: 3 * i + 3], axis=1)
    # print(new_dataset.shape)
    # print(y.shape)
    # sample2 = np.concatenate((new_dataset, y), axis=1)
    # np.save('new_clear_data_for_norm.npy', sample2)

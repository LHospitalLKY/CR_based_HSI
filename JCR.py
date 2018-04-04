# !/usr/bin/python2
# -*-coding: utf-8 -*-
"""
算法来源：Joint Within-Class Collaborative Representation for Hyperspectral Image Classification
Paper Author: Wei Li, Qian DU
Alg Author: Kaiyu Lei
输入：高光谱数据 -- mat格式的三维矩阵
输出：实验文件，包括：
        分类精确率表格 -- 四种JCR算法对应lambda的分类精确率，
        时间花费表格 -- 四种JCR算法对应lambda的时间花费
        分类图 -- 每一种类别定义一种颜色，将高光谱中每个像素点都用这种对应的颜色画出来

步骤：
	-- 导入mat数据，使之成为python能够识别的数据
	-- loop
	    -- 按比例选择训练集、开发集和测试集，对该类数据，一般测试集的数量要远远大于测试集，开发集从训练集中按一定比例选择选择
		-- loop
			-- 选定一个lambda，依次执行JCR1、JCR2、JCR3、JCR4，记录时间分类精确度与时间花费，绘出分类图
		-- end loop
	-- end loop
"""

## 导入的包
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from JCRC_Computing import Computing


## 导入数据文件
def import_data():
	'''
	导入高光谱数据矩阵：将一个.mat文件导入python
	:return:
	data_base -- 一个字典型变量，包含两个部分：
					data_base['sample'] -- 高光谱三维矩阵
					data_base['label] -- 高光谱中每个像素的类别
	'''
	data_base = {}
	dataFile1 = raw_input("输入样本文件名：")
	data1 = scipy.io.loadmat(dataFile1)
	sample = data1[sorted(data1.keys())[3]]     # 这里的下标可能随着不同文件而不同
	# print(sample)
	dataFile2 = raw_input("输入样本标签文件：")
	data2 = scipy.io.loadmat(dataFile2)
	label = data2[sorted(data2.keys())[3]]

	assert(sample.shape[0] == label.shape[0])

	data_base['sample'] = sample
	data_base['label'] = label

	return data_base


## 记录每类样本点的位置
def position(data_base):
	'''
	根据高光谱数据集信息得出每类样本点在map中的位置
	:param data_base: 输入样本数据与类别信息，是一个字典型变量
	:return: info_position_label，字典型变量，每类样本点在map中的位置
	'''
	assert(data_base.keys() == ['sample', 'label'])
	assert(len(data_base['sample'].shape) == 3)
	assert(len(data_base['label'].shape) == 2)
	assert(data_base['sample'].shape[0] == data_base['label'].shape[0])
	assert(data_base['sample'].shape[1] == data_base['label'].shape[1])

	info_position_label = {}

	label = data_base['label']

	for i in range(label.min(), label.max()):
		info_position_label['Position' + str(i)] = np.where(label == i)

	return info_position_label


## 按比例选择训练集、开发集和测试集
def train_select(data_base, info_position_label, rate1 = 0.3, rate2 = 0.0):
	'''
	根据比例选择训练集，每个类别都选取一定比例的样本作为训练集，从训练集中选取一定比例的样本作为dev集，剩余的作为测试集
	与此同时舍弃背景点
	:param rate1: 选取测试集的比例
	:param rate2: 选取dev集的比例
	:param data_base: 样本数据集
	:param info_position_label: 各类样本点位置信息
	:return:data_set，字典型变量，包括每一类的训练集和测试集，其格式为
						data_set['Train1']: 第1类样本的训练集
						data_set['Dev1]: 第1类样本的dev集
						data_set['Test1']: 第1类样本的测试集
	'''
	# TODO: 增加assert()段
	data_set = {}

	sample = data_base['sample']

	for i in range(len(info_position_label) - 1):
		# 使用cache保存当前类别所有样本集，m保存当前类的长度，使用random.sample方法来选取训练集
		cache_sample = sample[info_position_label['Position' + str(i + 1)]].tolist()
		m = len(info_position_label['Position' + str(i + 1)][0])

		data_set['Train' + str(i + 1)] = random.sample(cache_sample, int(rate1*m))

		cache_train = data_set['Train' + str(i + 1)]
		n = len(data_set['Train' + str(i + 1)])

		if rate2 != 0:
			data_set['Dev' + str(i + 1)] = random.sample(cache_train, int(rate2*n))


		# 使用列表推导式来得到训练集的index
		train_num =np.array([h for h in range(m)
		            for j in range(len(cache_train))
		                           if cache_train[j] == cache_sample[h]])

		test_num = np.array([k for k in range(m)
		                     if k not in train_num])

		data_set['Test' + str(i + 1)] = [x for x in cache_sample if cache_sample.index(x) not in train_num]
		data_set['Train' + str(i + 1) + 'Position'] = train_num
		data_set['Test' + str(i + 1) + 'Position']= test_num

	return data_set

## 在原图中增加空间近邻信息
def nearest_neighbors(map):
	'''
	对原数据进行处理，加入空间信息
	:param map: 原高光谱图片，三维矩阵，ndarray数据
	:return: map_neighbor 滤波处理之后的map，ndarray数据，大小同map相同
	'''
	[m, n, dim] = map.shape
	map_neighbor = np.zeros([m, n, dim])

	for i in range(1, m - 1):
		for j in range(1, n - 1):
			map_neighbor[i, j] = (map[i - 1, j - 1] + map[i - 1, j] + map[i - 1, j + 1] +
			                                 map[i, j - 1] + map[i, j] + map[i, j + 1]+
			                                 map[i + 1, j - 1] + map[i + 1, j] + map[i + 1, j + 1])/9

	for j in range(1, n - 1):
		map_neighbor[0, j] = (map[0, j - 1] + map[0, j] + map[0, j + 1] +
			                                 map[0, j - 1] + map[0, j] + map[0, j + 1]+
			                                 map[1, j - 1] + map[1, j] + map[1, j + 1])/9
		map_neighbor[m - 1, j] = (map[m - 2, j - 1] + map[m - 2, j] + map[m - 2, j + 1] +
			                                 map[m - 1, j - 1] + map[m - 1, j] + map[m - 1, j + 1]+
			                                 map[m - 1, j - 1] + map[m - 1, j] + map[m - 1, j + 1])/9

	for i in range(1, m - 1):
		map_neighbor[i, 0] = (map[i - 1, 0] + map[i - 1, 0] + map[i - 1, 1] +
			                                 map[i, 0] + map[i, 0] + map[i, 1]+
			                                 map[i + 1, 0] + map[i + 1, 0] + map[i + 1, 1])/9
		map_neighbor[i, n - 1] = (map[i - 1, n - 2] + map[i - 1, n - 1] + map[i - 1, n - 1] +
			                                 map[i, n - 2] + map[i, n - 1] + map[i, n - 1]+
			                                 map[i + 1, n - 2] + map[i + 1, n - 1] + map[i + 1, n - 1])/9

	map_neighbor[0, 0, :] = (2 * (map[0, 1] + map[1, 0] + map[1, 1]) + map[0, 0])/9
	map_neighbor[0, n - 1, :] = (2 * (map[0, n - 2] + map[1, n - 2] + map[1, n - 1]) + map[0, n - 1])/9
	map_neighbor[m - 1, 0, :] = (2 * (map[m - 2, 0] + map[m - 2, 1] + map[m - 1, 1]) + map[m - 1, 0])/9
	map_neighbor[m - 1, n - 1, :] = (2 * (map[m -2 , n - 2] + map[m - 1, n - 2] + map[m - 2, n - 1]) + map[m - 1, n - 1])/9

	return map_neighbor


## 选择使用的JRC模型
def model():
	model = raw_input('请输入使用的方法：')

	assert((model == 'JN_Test_Only' or model == 'JN_Test_Train'
	        or model == 'JR_Terms' or model == 'Smooth_Constraint'))

	return model

## 实验模块
# step1: 选取一个测试集
# step2: 将训练集按类别编成字典，循环使用这些字典来计算残差向量
#   step3: 比较这些残差项，得出测试集的类别
#   step4: 计算分类准确率

### Main函数
def main():
	data_base = import_data()
	data_base['sample'] = nearest_neighbors(data_base['sample'])
	info_position_label = position(data_base)
	data_set = train_select(data_base, info_position_label, 0.3, 0.2)

	## 进行实验
	for test_num in range(data_base['label'].min() + 1, data_base['label'].max()):

		# 选择测试集
		test = np.array(data_set['Test' + str(test_num)]).T

		Residual = {}
		file=open('Residual_of_' +str(test_num) + '.txt','w')
		for num_train in range(data_base['label'].min() + 1, data_base['label'].max()):
			train = np.array(data_set['Train' + str(num_train)]).T
			# TODO: 参数选择
			JRC = Computing(test, train, 0.1)

			# print JRC.alpha
			file.write(str(JRC.residual) + '\n')

			Residual['Residual of Train' + str(num_train)] = JRC.residual

		file.close()

		print('over1')

	print('Over')


# 执行JCR.py
if __name__ == '__main__':
	main()

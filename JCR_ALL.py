# !/usr/bin/python2
# -*-coding: utf-8 -*-
# 本文件是选取固定数量的样本点，将所有的训练点编为字典来进行运算
import _random
import numpy as np
from JCR_Separate import *
from JCRC_Computing import *

def train_select_constant(data_base, info_position_label, constant = 100):
	'''
	Select train set which contains same number of point by this function

	:param  data_base: 样本数据集，是一个dict类型的数据，分为sample和label，sample中存储map，label为对应的标签
			info_position_label: 样本标签信息，dict类型数据，记录了data_set中每个样本在原map中的位置
			constant: 整数型常量，表示每一类样本中选取多少个作为训练集
	:return:

	'''
	# 未被舍弃的样本计数器
	p = 0
	# 得到map
	sample = data_base['sample']
	# 返回dict初始化
	data_set ={}

	for i in range(len(info_position_label) - 1):

		m = len(info_position_label['Position' + str(i + 1)][0])

		if m < constant:
			continue
		else:
			p += 1
			cache_sample = sample[info_position_label['Position' + str(i + 1)]].tolist()

			# 获取训练集
			data_set['Train' + str(p)] = random.sample(cache_sample, constant)

			# 获取测试集
			n = len(data_set['Train' + str(p)])
			cache_train = data_set['Train' + str(p)]
			train_num = np.array([h for h in range(m)
			                        for j in range(len(cache_train))
			                            if cache_train[j] == cache_sample[h]])
			test_num = np.array([k for k in range(m)
		                            if k not in train_num])
			data_set['Test' + str(p)] = [x for x in cache_sample if cache_sample.index(x) not in train_num]

	return data_set, p


def main():
	data_base = import_data()
	data_base['sample'] = nearest_neighbors(data_base['sample'])
	info_position_label = position(data_base)
	constant = int(raw_input("请输入每一类训练集的个数："))
	data_set, p = train_select_constant(data_base, info_position_label, constant)


	# 合并训练集，并标记每个训练集结束的位置
	train_all = np.array(data_set['Train' + str(1)])
	for i in range(1, p):
		train_all = np.concatenate((train_all, data_set['Train' + str(i + 1)]), axis = 0)

	train_all = train_all.T

	# debug
	# print train_all
	Residual = {}
	for test_l in range(p):
		# 选取测试集
		test = np.array(data_set['Test' + str(test_l + 1)]).T

		Residual_array = np.zeros([p, np.shape(test)[1]])

		for l in range(np.shape(test)[1]):

			## 训练点分类
			for train_l in range(p):
				train = np.array(data_set['Train' + str(train_l + 1)]).T
				JRC = Computing(test[:, l], train, 0.5)

				# print JRC.alpha
				# print('Done!')
				Residual_array[train_l, l] = JRC.residual

		fuck = np.where(Residual_array == np.amin(Residual_array, axis = 0))[0]     # 找到分到的类别
		fuck = (fuck == test_l)
		num_right = float(list(fuck).count(1))
		accuracy = num_right/float(len(fuck))

		print('Test' + str(test_l + 1) + ' num = ' + str(len(fuck)))
		print('Test' + str(test_l + 1) + ' accuracy = ' + str(accuracy))


	print('Over!')


if __name__ == '__main__':
	main()

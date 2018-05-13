# !/usr/bin/python2
# -*-coding: utf-8 -*-
# 本文件是选取固定数量的样本点，将所有的训练点编为字典来进行运算
import _random
import numpy as np
from JCR_Separate import *
from JCRC_Computing import *
from pJCR_Computing import *

def train_select_constant(data_base, info_position_label, constant = 100):
	"""
	Select train set which contains same number of point by this function

	:param  data_base: 样本数据集，是一个dict类型的数据，分为sample和label，sample中存储map，label为对应的标签
			info_position_label: 样本标签信息，dict类型数据，记录了data_set中每个样本在原map中的位置
			constant: 整数型常量，表示每一类样本中选取多少个作为训练集
	:return:p: 选择之后的样本数量

	"""
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
			cache_train = data_set['Train' + str(p)]
			# 加入训练点的位置信息
			num = 0
			data_set['Train' + str(p) + 'Position'] = []
			for train_sample in cache_train:
				indexInInfoPos = cache_sample.index(train_sample)
				dp = np.array([info_position_label['Position' + str(i + 1)][0][indexInInfoPos],
							   info_position_label['Position' + str(i + 1)][0][indexInInfoPos]])
				data_set['Train' + str(p) + 'Position'].append(dp)
				num += 1
			del num
			del cache_train

			# 获取测试集
			n = len(data_set['Train' + str(p)])
			cache_train = data_set['Train' + str(p)]
			train_num = np.array([h for h in range(m)
			                        for j in range(len(cache_train))
			                            if cache_train[j] == cache_sample[h]])
			test_num = np.array([k for k in range(m)
		                            if k not in train_num])
			data_set['Test' + str(p)] = [x for x in cache_sample if cache_sample.index(x) not in train_num]
			cache_test = data_set['Test' + str(p)]
			# 加入测试点的位置信息
			num = 0
			data_set['Test' + str(p) + 'Position'] = []
			for test_sample in cache_test:
				indexInInfoPos = cache_sample.index(test_sample)
				dp = np.array([info_position_label['Position' + str(i + 1)][0][indexInInfoPos],
							   info_position_label['Position' + str(i + 1)][1][indexInInfoPos]])
				data_set['Test' + str(p) + 'Position'].append(dp)
				num += 1
				# print(data_base['label'][dp[0]][dp[1]])
			del num
			del cache_test

	return data_set, p


def experiment_model(conNum = False, method = 'JCR', savePreMatrix = False):
	"""
	实验模块，输入参数为开关
	:param data_base: 使用的数据集
	:param conNum: 训练集是否选取定值
	:param method: 选择模型
	:param savePreMatrix: 是否保存分类矩阵（需要作图时保存，不做图时可以舍弃）
	:return:
	"""
	data_base = import_data()
	data_base['sample'] = nearest_neighbors(data_base['sample'])
	info_position_label = position(data_base)
	predictGraphic = data_base['label']	# 初始化predictGraphic

	# 输入正则项系数lambd
	lambd = input('请输入lambd:')
	lambd = float(lambd)

	# 十次实验取平均值
	accuracy = []
	for expNum in range(2):
		# 训练集选取方式
		if conNum == True:
			if expNum == 0:
				global constant
				constant = int(input("请输入每一类训练集的个数："))
			data_set, p = train_select_constant(data_base, info_position_label, constant)
		else:
			data_set = train_select_propatation(data_base, info_position_label, rate1 = 0.1)
			p = len(info_position_label) - 1

		# 模型选择
		if method == 'pJCR':
			if expNum == 0:
				global pNorm
				pNorm = input('请输入lp的值（1 < p < 2）：')
				pNorm = float(pNorm)
				fileName = 'expResult/' + method + '/_' + str(lambd) + '_' + str(pNorm) + '.cvs'
		elif method == 'JCR':
			if expNum == 0:
				fileName = 'expResult/' + method + '/_' + str(lambd) + '.cvs'

		# 合并训练集，并标记每个训练集结束的位置
		train_all = np.array(data_set['Train' + str(1)])
		for i in range(1, p):
			train_all = np.concatenate((train_all, data_set['Train' + str(i + 1)]), axis = 0)

		train_all = train_all.T

		# debug
		# print train_all
		Residual = {}

		accTmp = np.zeros(p)
		for test_l in range(p):
			# 选取测试集
			test = np.array(data_set['Test' + str(test_l + 1)]).T
			testPosition = data_set['Test' + str(test_l + 1)]
			Residual_array = np.zeros([p, np.shape(test)[1]])

			for l in range(np.shape(test)[1]):
				## 训练点分类
				for train_l in range(p):
					train = np.array(data_set['Train' + str(train_l + 1)]).T

					if method == 'JCR':
						JCR = Computing(test[:, l], train, lambd)
					elif method == 'pJCR':
						JCR = pJCR_Computing(test[:, l], train, lambd, pNorm)

					# print JRC.alpha
					# print('Done!')
					Residual_array[train_l, l] = JCR.residual

			del l
			fuck = np.where(Residual_array == np.amin(Residual_array, axis = 0))[0]     # 找到分到的类别
			# 将分好的类对应到原图中的位置上
			for l in range(np.shape(test)[1]):
				axis_1 = data_set['Test' + str(test_l + 1) +'Position'][l][0]
				axis_2 = data_set['Test' + str(test_l + 1) +'Position'][l][1]
				predictGraphic[axis_1][axis_2] = fuck[l] + 1

			fuck = (fuck == test_l)
			num_right = float(list(fuck).count(1))
			accTmp[test_l] = num_right/float(len(fuck))

			print('Test' + str(test_l + 1) + ' num = ' + str(len(fuck)))
			print('Test' + str(test_l + 1) + ' accuracy = ' + str(accTmp[test_l]))

		accuracy.append(accTmp)
		print('Over!')

	# 如果需要保存分类后的类别矩阵
	if savePreMatrix == True:
		np.savetxt('predictMatrix.cvs', predictGraphic, delimiter = ',')
		np.savetxt(fileName, accuracy, delimiter = ',')
		return accuracy, predictGraphic
	else:
		del predictGraphic
		np.savetxt(fileName, accuracy, delimiter=',')
		return accuracy


def main():
	accuracy = experiment_model(conNum = False, method = 'JCR', savePreMatrix = False)
	"""
	method = 'pJCR'
	lambd = '0.5'
	p = '1.5'
	fileName = 'expResult/' + method + '/_' + str(lambd) +'_' + str(p) + '.cvs'
	k = np.zeros(10)
	np.savetxt(fileName, k, delimiter = ',')
	"""

if __name__ == '__main__':
	main()

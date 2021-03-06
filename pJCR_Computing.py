# -*-coding: utf-8 -*-
# 本文件是pJCR计算模块，相比JCR，主要将正则项的阶数改为p阶

import numpy as np
import math

class pJCR_Computing:
	'''
	类属性：
	y: 待测样本，p维向量，其中p是光谱长度
	X: 字典集合，pxn维矩阵，其中n为字典的中样本数量，p为光谱长度
	lambd: 正则项参数
	Tikhonov: Tikhonov矩阵，是一个对角矩阵，其中对角线的内容是待测样本和字典中所有样本的残差
	alpha: 线性表达系数，维度为n
	residual: 待测样本与字典集合的残差
	'''
	n = 1
	y = []
	X = []

	lambd = .0
	p = 2.

	__Tikhonov = []
	alpha = []
	residual = []

	# 初始化
	def __init__(self, y, X, lambd, p = 1.):

		self.y = y
		self.X = X
		self.lambd = lambd
		self.p = p

		dim_x = X.shape
		dim_y = y.shape

		self.__Tikhonov = np.zeros([dim_x[1], dim_x[1]])
		# self.residual = np.zeros(dim_y[1])

		## 计算Tiknonov矩阵
		if len(dim_y) == 1:   # 如果y是一个向量
			self.Tikhonov(self.y, self.X)
		else:               # 如果y是一个矩阵
			cache_Ti = self.__Tikhonov
			for n in range(dim_y[1]):
				# print y[:, n]     # Debug使用
				cache_Ti = cache_Ti + self.Tikhonov(self.y[:, n], self.X)
			self.__Tikhonov = (1./dim_y[1]) * cache_Ti

		self.Alpha(self.y, self.X, self.__Tikhonov, self.lambd, p)

		## 计算残差
		if len(dim_y) == 1:
			self.Residual(self.y, self.X, self.alpha)
		else:
			cache_Re = self.residual
			for n in range(dim_y[1]):
				cache_Re.append(self.Residual(self.y[:, n], self.X, self.alpha[:, n]))
			self.residual = cache_Re

		# self.print_Ti()
		# print 'Done!'


	def print_Ti(self):
		print(self.__Tikhonov)


	def Tikhonov(self, y, X):
		# TODO: 添加维度检查
		p = X.shape[0]
		n = X.shape[1]

		T_cache = np.zeros(n)

		for i in range(n):
			T_cache[i] = math.sqrt(np.dot((y - X[:, i]).T, (y - X[:, i])))

		self.__Tikhonov = np.diag(T_cache)

		return self.__Tikhonov


	def Alpha(self, y, X, Tikhonov, lambd, p = 1.):

		# 循环标记
		# print y.shape
		k = 0
		erro = 1
		beta_0 = np.ones([X.shape[1], ])

		while (k <= 50 and erro > 0.001):
			# 计算本次alpha
			k += 1
			B = np.diag(beta_0)
			B = B ** 2
			Q = np.linalg.inv((np.dot(X.T, X) + lambd*np.dot(B, np.dot(Tikhonov.T, Tikhonov))))
			U = np.dot(Q, X.T)
			self.alpha = np.dot(U, y)

			# 更新erro与beta_0
			beta_1 = p * (abs(self.alpha) + 0.02 * np.ones(self.alpha.shape)) ** (p - 2)
			beta_1 = beta_1.reshape(beta_1.shape[0], )
			erro = np.min(abs(beta_1 - beta_0))
			beta_0 = beta_1


		# self.alpha = np.dot(U, y)
		assert(self.alpha.shape[0] == X.shape[1])



	def Residual(self, y, X, alpha):
		assert(X.shape[0] == y.shape[0])
		assert(alpha.shape[0] == X.shape[1])
		R_cache = y - np.dot(X, alpha)
		self.residual = math.sqrt(np.dot(R_cache.T, R_cache))/math.sqrt(np.dot(alpha.T, alpha)) # 使用相对残差时加 '/math.sqrt(np.dot(alpha.T, alpha))'

		return self.residual


# 单元测试用例
if __name__ == '__main__':
	y = np.random.rand(20, )   #
	X = np.random.rand(20, 40)

	print(y.shape)
	print(X.shape)

	JRC = pJCR_Computing(y, X, 0.1, 1.1)

	print(JRC.alpha)
	print(JRC.residual)

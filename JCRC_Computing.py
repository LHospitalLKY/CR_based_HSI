# -*-coding: utf-8 -*-
# 本文件是JRC四个算法的计算模块
import numpy as np
import math

class Computing:
	'''
	类属性：
	y: 待测样本，p维向量，其中p是光谱长度
	X: 字典集合，pxn维矩阵，其中n为字典的中样本数量，p为光谱长度
	lambd: 正则项参数
	Tikhonov: Tikhonov矩阵，是一个对角矩阵，其中对角线的内容是待测样本和字典中所有样本的残差
	alpha: 线性表达系数，维度为n
	residual: 待测样本与字典集合的残差
	'''
	y = []
	X = []
	lambd = .0
	__Tikhonov = []
	alpha = []
	residual = 0

	# 初始化
	def __init__(self, y, X, lambd = .0):

		assert(X.shape[0] == y.shape[0])

		self.y = y
		self.X = X
		self.lambd = lambd

		self.Tikhonov(y, X)

		self.Alpha(self.y, self.X, self.__Tikhonov, self.lambd)
		self.Residual(self.y, self.X, self.alpha)

		self.print_Ti()
		print 'Done!'


	def print_Ti(self):
		print self.__Tikhonov


	def Tikhonov(self, y, X):
		# TODO: 添加维度检查
		p = X.shape[0]
		n = X.shape[1]

		T_cache = np.zeros(n)

		for i in range(n):
			T_cache[i] = math.sqrt(np.dot((y - X[:, i]).T, (y - X[:, i])))

		self.__Tikhonov = np.diag(T_cache)

		return self.__Tikhonov


	def Alpha(self, y, X, Tikhonov, lambd):
		Q = np.linalg.inv((np.dot(X.T, X) + lambd*np.dot(Tikhonov.T, Tikhonov)))
		U = np.dot(Q, X.T)
		self.alpha = np.dot(U, y)
		assert(self.alpha.shape[0] == X.shape[1])


	def Residual(self, y, X, alpha):
		assert(X.shape[0] == y.shape[0])
		assert(alpha.shape[0] == X.shape[1])
		R_cache = np.dot(X, alpha) - y
		self.residual = math.sqrt(np.dot(R_cache.T, R_cache))


# 单元测试用例
if __name__ == '__main__':
	y = np.random.rand(20)
	X = np.random.rand(20, 40)

	print y.shape
	print X.shape

	JRC = Computing(y, X, 0.1)


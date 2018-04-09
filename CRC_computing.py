# -*-coding: utf-8 -*-
import numpy as np
import math

class CRC_Computing:
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
	alpha = []
	residual = []

	# 初始化
	def __init__(self, y, X, lambd = .0):

		assert(X.shape[0] == y.shape[0])

		self.y = y
		self.X = X
		self.lambd = lambd

		dim_x = X.shape
		dim_y = y.shape

		self.residual = np.zeros(dim_y[1])

		self.Alpha(self.y, self.X, self.lambd)

		## 计算残差
		if len(dim_y) == 1:
			self.Residual(self.y, self.X, self.alpha)
		else:
			cache_Re = self.residual
			for n in range(dim_y[1]):
				cache_Re[n] = self.Residual(self.y[:, n], self.X, self.alpha[:, n])
			self.residual = cache_Re

		# self.print_Ti()
		# print 'Done!'

	def Alpha(self, y, X, lambd):
		A = np.dot(X.T, X)
		n = A.shape[0]
		Q = np.linalg.inv((np.dot(X.T, X) + lambd*np.eye(n)))
		U = np.dot(Q, X.T)
		self.alpha = np.dot(U, y)
		assert(self.alpha.shape[0] == X.shape[1])


	def Residual(self, y, X, alpha):
		assert(X.shape[0] == y.shape[0])
		assert(alpha.shape[0] == X.shape[1])
		R_cache = np.dot(X, alpha) - y
		self.residual = math.sqrt(np.dot(R_cache.T, R_cache))  # 使用相对残差时加 '/math.sqrt(np.dot(alpha.T, alpha))'

		return self.residual


# 单元测试用例
if __name__ == '__main__':
	y = np.random.rand(20, 50)   #
	X = np.random.rand(20, 40)

	print y.shape
	print X.shape

	CRC = CRC_Computing(y, X, 0.1)

	print CRC.alpha
	print CRC.residual

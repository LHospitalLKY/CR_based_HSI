# -*-coding: utf-8 -*-
import numpy as np
import math

class pCRC_Computing:
	'''
	类属性：
	y: 待测样本，d维向量，其中d是光谱长度
	X: 字典集合，pxn维矩阵，其中n为字典的中样本数量，p为光谱长度
	lambd: 正则项参数
	p: 表示使用的模
	alpha: 线性表达系数，维度为n
	residual: 待测样本与字典集合的残差
	'''
	y = []
	X = []
	lambd = .0
	p = 1.
	alpha = []
	residual = []

	# 初始化
	def __init__(self, y, X, lambd = .0, p = 1.):

		assert(X.shape[0] == y.shape[0])

		self.y = y
		self.X = X
		self.lambd = lambd
		self.p = p

		dim_x = X.shape
		dim_y = y.shape

		self.residual = np.zeros(dim_y[1])

		self.Alpha(self.y, self.X, self.lambd, self.p)

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

	def Alpha(self, y, X, lambd, p):
		# 循环变量
		k = 0
		erro = 1
		beta_0 = np.ones([X.shape[1], ])

		while (k <= 200 and erro > 0.00001):
			# 计算本次alpha
			k += 1
			B = np.diag(beta_0)
			B = B ** 2
			Q = np.linalg.inv((np.dot(X.T, X) + lambd*B))
			U = np.dot(Q, X.T)
			self.alpha = np.dot(U, y)

			# 更新erro与beta_0
			beta_1 = p * (abs(self.alpha) + 0.02 * np.ones(self.alpha.shape)) ** (p - 2)
			beta_1 = beta_1.reshape(beta_1.shape[0], )
			erro = np.min(abs(beta_1 - beta_0))
			beta_0 = beta_1

		assert(self.alpha.shape[0] == X.shape[1])


	def Residual(self, y, X, alpha):
		assert(X.shape[0] == y.shape[0])
		assert(alpha.shape[0] == X.shape[1])
		R_cache = np.dot(X, alpha) - y
		self.residual = math.sqrt(np.dot(R_cache.T, R_cache))  # 使用相对残差时加 '/math.sqrt(np.dot(alpha.T, alpha))'

		return self.residual


# 单元测试用例
if __name__ == '__main__':
	y = np.random.rand(20, 1)   #
	X = np.random.rand(20, 40)

	print y.shape
	print X.shape

	pCRC = pCRC_Computing(y, X, 0.1)

	print pCRC.alpha
	print pCRC.residual

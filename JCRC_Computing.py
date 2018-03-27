# -*-coding: utf-8 -*-
# 本文件是JRC四个算法的计算模块

class JRC:
	# 定义基本属性
	map = {}
	data_set = {}
	method = 'JN_Test_Only'
	neighbor_data = 0
	Tikhonov =

	# 初始化
	def __init__(self, map, data_set, model):
		map = map
		data_set = data_set
		method = model

		# 将原数据转变为带近邻信息的数据
		map_neighbor = self.__nearest_neighbors(map)

		if
		Tikhonov =


	# 增加近邻空间信息
	def __nearest_neighbors(self, map):
		return 0

	def __Tikhonov(self, y, X, model):
		return 0

	def Alpha(self, y, X, Tikhonov, lamb):
		return 0

	def Residual(self, alpha, X, y):
		return 0

	def Class(self, X, alpha, y):
		return 0


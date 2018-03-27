import numpy as np


def nearest_neighbors(map):
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


map = np.random.rand(10, 10, 20)
map_neighbor = np.zeros([10, 10, 20])

nearest_neighbors(map)
print(nearest_neighbors(map))

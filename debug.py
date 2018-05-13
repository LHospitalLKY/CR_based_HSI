import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from JCR_Separate import *

Z = np.random.randn(100, 100)

# 源数据
data_base = import_data()
tmp = data_base['label']
origMatrix = np.zeros([tmp.shape[0], tmp.shape[1]])
for i in range(tmp.shape[0]):
	for j in range(tmp.shape[1]):
		origMatrix[i][j] = np.float(tmp[i][j])
origGraph = plt.subplot(121)
origGraph.imshow(origMatrix, cmap = cm.RdYlGn)

# 预测数据
a = np.loadtxt("/home/lhospital/MyProgramm/CR_based_HSI/predictMatrix.cvs", dtype = np.str, delimiter=",")

preMatrix = np.zeros([a.shape[0], a.shape[1]])

for i in range(a.shape[0]):
	for j in range(a.shape[1]):
		preMatrix[i][j] = np.float(a[i][j])

preGraph = plt.subplot(122)
preGraph.imshow(preMatrix, cmap = cm.RdYlGn)

plt.show()

a = origMatrix == preMatrix

print(a)
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('./test.txt')
n = len(data)
x_axis = np.arange(n)

plt.plot(x_axis, data)
plt.savefig('./plot.png')
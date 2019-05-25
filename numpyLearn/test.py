import numpy as np


a = np.array([[1, 2, 1], [2, 2, 2]])
print(a)
print(a.ndim)
print(a.shape)
print(a.dtype)
print(a.size)
print(a.itemsize)
print(type(a))

z = np.random.randn(3, 3, 3)
print(z)

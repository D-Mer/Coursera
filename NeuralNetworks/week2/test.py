import numpy as np

A = np.array([[1, 2, 3]])
B = np.zeros(A.shape)

B[A > 2] = 1
print(B)

import numpy as np

a = [[[0]*5]*5]*2
a = np.array(a)
a =a.tolist()
a[0][0][0] = 1
a = np.array([[1,2],[3,4]])
b = np.array([[4,5],[6,7]])
c = np.stack([a,b],axis=1)
print(c)
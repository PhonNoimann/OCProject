import numpy as np
import matplotlib.pyplot as plt

def feasPlot( A, B, X):
    for i in range(len(A)):
        t = np.arange(1,len(A),1)
        t = np.delete(t, i)
        z = np.linalg.solve(A[t],B[t])
        temp = A[i,0]*z[0]+A[i,1]*z[1]-B[i]
        if temp <= 0:
            X.append(z)
    print(X)

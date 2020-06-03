from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def [fig1, ax1] = feasPlot( A, B, x1, x2):
    for i in range(len(A)):
        t = np.arange(0,len(A),1)
        t = np.delete(t, i)
        print(t)
        z = np.linalg.solve(A[t],B[t])
        temp = A[i,0]*z[0]+A[i,1]*z[1]-B[i]
        if temp <= 0:
            x1.append(z[0])
            x2.append(z[1])
    for j in range(1):
        for i in range(len(A)):
            z = np.array([0.,0.])
            z[j] = B[i]/A[i,j]
            temp = A[0:,0]*z[0]+A[0:,1]*z[1]-B
            if temp[0] <= 0 and temp[1] <= 0 and temp[2] <= 0:
                x1.append(z[0])
                x2.append(z[1])
    fig1, ax1 = plt.subplots()
    s = np.argsort(x1)
    s = np.append(s,0)
    x1 = np.array(x1)
    x2 = np.array(x2)
    ax1.plot(x1[s],x2[s],'.k')
    ax1.plot(x1[s],x2[s],'k'),linewi
    ax1.set_title('Feasible region')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.grid(True)
    #plt.savefig('test.png')
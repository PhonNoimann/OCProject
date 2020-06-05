from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def feasPlot( A, B, x1, x2):
    Feas1 = [] + x1
    Feas2 = [] + x2
    for i in range(len(A)):
        t = np.arange(0,len(A),1)
        t = np.delete(t, i)
        x = np.linalg.solve(A[t],B[t])
        temp = A[i,0]*x[0]+A[i,1]*x[1]-B[i]
        if temp <= 0:
            Feas1 = np.append(Feas1, x[0])
            Feas2 = np.append(Feas2, x[1])
    for j in range(1):
        for i in range(len(A)):
            x = np.array([0.,0.])
            x[j] = B[i]/A[i,j]
            temp = A[0:,0]*x[0]+A[0:,1]*x[1]-B
            if temp[0] <= 0 and temp[1] <= 0 and temp[2] <= 0:
                Feas1 = np.append(Feas1, x[0])
                Feas2 = np.append(Feas2, x[1])
    fig1, ax1 = plt.subplots()
    s = np.argsort(Feas1)
    s = np.append(s,0)
    Feas1 = np.array(Feas1)
    Feas2 = np.array(Feas2)
    ax1.plot(Feas1[s],Feas2[s],'*k')
    ax1.plot(Feas1[s],Feas2[s],'k', linewidth=1.5, label = 'Feasible region')
    ax1.set_title('Solution to bi-objective linear optimization problem using weighting method')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.grid(True)
    plt.savefig('test.png')

    return (fig1, ax1)
  
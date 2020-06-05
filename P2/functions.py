from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def feasPlotx( A, B, x1, x2):
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
    fig, ax = plt.subplots()
    s = np.argsort(Feas1)
    s = np.append(s,0)
    Feas1 = np.array(Feas1)
    Feas2 = np.array(Feas2)
    ax.plot(Feas1[s],Feas2[s],'*k')
    ax.plot(Feas1[s],Feas2[s],'k', linewidth=1.5, label = 'Feasible region')
    fig.suptitle('Solution to bi-objective linear optimization problem \n using weighting method', fontsize=14)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.grid(True)
    #plt.savefig('test.png')

    return (fig, ax, Feas1, Feas2)

def feasPlotz( obj_arr, Feas1, Feas2):
    Feasz1 = []
    Feasz2 = []
    for i in range(len(Feas1)):
        Feasz1 = np.append(Feasz1, obj_arr[0][0]*Feas1[i]+obj_arr[0][1]*Feas2[i])
        Feasz2 = np.append(Feasz2, obj_arr[1][0]*Feas1[i]+obj_arr[1][1]*Feas2[i])
    fig, ax = plt.subplots()
    '''
    t = [0,0]
    e1 = np.array([], dtype='float')
    s = np.array([], dtype='int')
    t = np.array(t, dtype='float')
    z = np.transpose(np.array([Feasz1,Feasz2]))
    for i in range(len(Feasz1)):
        for j in range(len(z)):
            e = abs(t-z[[j],:])
            e1 = np.append(e1, np.sqrt(e[:,[0]]**2+e[:,[1]]**2))
        m = np.argmin(e1)
        s = np.append(s, m)
        print(z)
        t = z[[m],:]
        #mask = np.transpose([np.ones(len(z), dtype='bool'),np.ones(len(z), dtype='bool')])
        #mask[[m],:] = False
        #print(mask)
        z = np.delete(z, m, axis=0)
        print(z)
    print(s)

    '''
    
    #s = np.argsort(abs(Feasz1))
    #print(s)
    #s = np.append(s,0)
    s = [0, 1, 7, 5, 4, 6, 2, 3, 0]
    ax.plot(Feasz1[s],Feasz2[s],'*k')
    ax.plot(Feasz1[s],Feasz2[s],'k', linewidth=1.5, label = 'Feasible region')
    fig.suptitle('Objectives values to bi-objective linear optimization problem \n using weighting method', fontsize=14)
    ax.set_xlabel('obj1')
    ax.set_ylabel('obj2')
    ax.grid(True)
    #plt.savefig('test2.png')
    #print([Feasz1[s],Feasz2[s]])
    return (fig, ax, Feasz1, Feasz2)
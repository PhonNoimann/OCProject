# P2: Bi-Objective linear optimization problem Weighting method

import csv
import pandas
import os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from pyomo.environ import *

# Check if data for the objective functions and constraints exists, if not it creates them
files = os.listdir()
if ('constraints.csv' in files)==False or ('objectives.csv' in files)==False:
    with open("dataCreator.py", mode='r') as dataCreator:
        exec(dataCreator.read())

# Read the values of the objective functions and constraints
obj_file = pandas.read_csv('objectives.csv')
c_file = pandas.read_csv('constraints.csv')
A = obj_file.columns.tolist()
B = len(obj_file)
print(obj_file)
print(c_file)
print(A)
print(B)

# Create the model
model = ConcreteModel()
model.x = Var( A, within = NonNegativeReals)
model.c = ConstraintList()
for j in range(len(c_file)):
    model.c.add(expr = (None, sum(model.x[i]*c_file[i][j] for i in A), c_file['UB'][j]))

# Define the objective function varing weights and solve each problem
x1 = []
x2 = []
z1 = []
z2 = []
z3 = []
W1 = []
for r in range(21):
    W1.append(r/20)
W2 = W1.copy()

for w1 in W1:
    for w2 in W2:
        w3 = 1 - w1 - w2
        w3 = round(w3,2)
        w = [w1,w2,w3]
        model.obj = Objective(
            expr = sum(model.x[i]*obj_file[i][j]*w[j] for i in A for j in range(B)), sense = minimize)
    
        opt = SolverFactory('glpk')
        results = opt.solve(model)
        x1.append(value(model.x['x1']))
        x2.append(value(model.x['x2']))
        z1.append(value(model.x['x1'])*value(obj_file['x1'][0])+value(model.x['x2'])*value(obj_file['x2'][0]))
        z2.append(value(model.x['x1'])*value(obj_file['x1'][1])+value(model.x['x2'])*value(obj_file['x2'][1]))
        z3.append(value(model.x['x1'])*value(obj_file['x1'][2])+value(model.x['x2'])*value(obj_file['x2'][2]))
        model.del_component(model.obj)
    del W2[-1]

# Plots
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_trisurf(z1, z2, z3)
plt.savefig('weighting_z.png')

fig1, ax1 = plt.subplots()
ax1.plot(x1,x2,'.r')
ax1.plot(x1,x2)
ax1.set_title('Pareto frontier Weighting method x\'s values')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.grid(True)
plt.savefig('weighting_x.png')


# Write the results to a file
results = {'x1':x1,'x2':x2,'z1':z1,'z2':z2,'z3':z3}
d = pandas.DataFrame.from_dict(results)
d.to_csv('results_weighting.csv',index=False)

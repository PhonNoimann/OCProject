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
        w3=round(w3,2)
        if w3 < 0:
            break
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

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_trisurf(z1, z2, z3)

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.savefig('weighting_z.png')


fig1, ax1 = plt.subplots()
ax1.plot(x1,x2,'.r')
ax1.plot(x1,x2)
ax1.set_title('Pareto frontier Weighting method x\'s values')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.grid(True)
plt.savefig('weighting_x.png')
"""
for x,y in zip(x1,x2):
    label = "(" + "{:.1f}".format(x) + "," + "{:.1f}".format(y) + ")"
    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(5,-5)) # distance from text to points (x,y)
plt.savefig('pareto_weighting_x.png')

fig2, ax2 = plt.subplots()
ax2.plot(z1,z2,'.r')
ax2.plot(z1,z2,)
ax2.set_title('Pareto frontier Weighting method z\'s values')
ax2.set_xlabel('z1')
ax2.set_ylabel('z2')
ax2.grid(True)

for x,y in zip(z1,z2):
    label = "(" + "{:.1f}".format(x) + "," + "{:.1f}".format(y) + ")"
    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,-5)) # distance from text to points (x,y)
plt.savefig('pareto_weighting_z.png')
"""

# Write the results of the Pareto Frontier to a file
with open('results_weighting.txt',mode='w') as results_file:
    results_file.write('Variable x\n\n')
    results_file.write('   x1 = ' + str(x1) + '\n\n')
    results_file.write('   x2 = ' + str(x2) + '\n\n')
    results_file.write('   z1 = ' + str(z1) + '\n\n')
    results_file.write('   z2 = ' + str(z2) + '\n\n')
    results_file.write('   z3 = ' + str(z3) + '\n')

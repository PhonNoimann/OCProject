# P2: Bi-Objective linear optimization problem Weighting method

import csv
import pandas
import os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import *
from functions import *

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
for z in np.linspace(0,1,5):
    w=[z,1-z]
    model.obj = Objective(
        expr = sum(model.x[i]*obj_file[i][j]*w[j] for i in A for j in range(B)), sense = minimize)
    
    opt = SolverFactory('glpk')
    results = opt.solve(model)
    x1.append(value(model.x['x1']))
    x2.append(value(model.x['x2']))
    z1.append(value(model.x['x1'])*value(obj_file['x1'][0])+value(model.x['x2'])*value(obj_file['x2'][0]))
    z2.append(value(model.x['x1'])*value(obj_file['x1'][1])+value(model.x['x2'])*value(obj_file['x2'][1]))
    model.del_component(model.obj)


c_arr = np.array(c_file)
a = np.array(c_arr[:,[0,1]], dtype='float')
b = np.array(c_arr[:,3], dtype='float')

obj_arr = np.array(obj_file)
fig1, ax1, Feas1, Feas2 = feasPlotx(a,b,x1,x2)


ax1.plot(x1,x2,'*r', label = 'Pareto solution')
ax1.plot(x1,x2,'b', label = 'Pareto frontier')
plt.legend()


label = "(" + "{:.1f}".format(x1[0]) + "," + "{:.1f}".format(x2[0]) + ")"
plt.annotate(label, # this is the text
             (x1[0],x2[0]), # this is the point to label
             textcoords="offset points", # how to position the text
             xytext=(5,7)) # distance from text to points (x,y)
label = "(" + "{:.1f}".format(x1[2]) + "," + "{:.1f}".format(x2[2]) + ")"
plt.annotate(label, # this is the text
             (x1[2],x2[2]), # this is the point to label
             textcoords="offset points", # how to position the text
             xytext=(5,1)) # distance from text to points (x,y)
label = "(" + "{:.1f}".format(x1[4]) + "," + "{:.1f}".format(x2[4]) + ")"
plt.annotate(label, # this is the text
             (x1[4],x2[4]), # this is the point to label
             textcoords="offset points", # how to position the text
             xytext=(-42,-10)) # distance from text to points (x,y)

plt.savefig('pareto_weighting_x.png')

fig2, ax2, Feasz1, Feasz2 = feasPlotz(obj_arr, Feas1, Feas2)

ax2.plot(z1,z2,'*r', label = 'Pareto solution')
ax2.plot(z1,z2,'b', label = 'Pareto frontier')
plt.legend()

label = "(" + "{:.1f}".format(z1[0]) + "," + "{:.1f}".format(z2[0]) + ")"
plt.annotate(label, # this is the text
             (z1[0],z2[0]), # this is the point to label
             textcoords="offset points", # how to position the text
             xytext=(-50,-5)) # distance from text to points (x,y)
label = "(" + "{:.1f}".format(z1[2]) + "," + "{:.1f}".format(z2[2]) + ")"
plt.annotate(label, # this is the text
             (z1[2],z2[2]), # this is the point to label
             textcoords="offset points", # how to position the text
             xytext=(5,0)) # distance from text to points (x,y)
label = "(" + "{:.1f}".format(z1[4]) + "," + "{:.1f}".format(z2[4]) + ")"
plt.annotate(label, # this is the text
             (z1[4],z2[4]), # this is the point to label
             textcoords="offset points", # how to position the text
             xytext=(5,-20)) # distance from text to points (x,y)

plt.savefig('pareto_weighting_z.png')

# Write the results of the Pareto Frontier to a file
with open('results.txt',mode='w') as results_file:
    results_file.write('Variable x\n\n')
    results_file.write('   x1 = ' + str(x1) + '\n\n')
    results_file.write('   x2 = ' + str(x2) + '\n\n')
    results_file.write('   z1 = ' + str(z1) + '\n\n')
    results_file.write('   z2 = ' + str(z2) + '\n')
    

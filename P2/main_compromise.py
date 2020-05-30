# P2: Bi-Objective linear optimization problem Compromise method

import csv
import pandas
import os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
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
for z in np.linspace(0,1,50):
    w=[z,1-z]
#    model.obj = Objective(
#        expr = sum(((model.x[i]*obj_file[i][j])**2)*w[j] for i in A for j in range(B)), sense = minimize)
    model.obj = Objective(
        expr = ((obj_file['x1'][0]*model.x['x1']+obj_file['x2'][0]*model.x['x2'])**4)*w[0] + ((obj_file['x1'][1]*model.x['x1']+obj_file['x2'][1]*model.x['x2'])**4)*w[1], sense = minimize)
    
    
    opt = SolverFactory('ipopt')
    results = opt.solve(model)
    x1.append(value(model.x['x1']))
    x2.append(value(model.x['x2']))
    z1.append(value(model.x['x1'])*value(obj_file['x1'][0])+value(model.x['x2'])*value(obj_file['x2'][0]))
    z2.append( value(model.x['x1']) * value(obj_file['x1'][1]) + value(model.x['x2']) * value(obj_file['x2'][1]) )
    model.pprint()
    model.del_component(model.obj)

fig1, ax1 = plt.subplots()
ax1.plot(x1,x2,'.r')
ax1.plot(x1,x2)
ax1.set_title('Pareto frontier Compromise method x\'s values')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.grid(True)

"""
for x,y in zip(x1,x2):
    label = "(" + "{:.1f}".format(x) + "," + "{:.1f}".format(y) + ")"
    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(5,-5)) # distance from text to points (x,y)
"""
plt.savefig('pareto_compromise_x.png')

fig2, ax2 = plt.subplots()
ax2.plot(z1,z2,'.r')
ax2.plot(z1,z2,)
ax2.set_title('Pareto frontier Weighting method z\'s values')
ax2.set_xlabel('z1')
ax2.set_ylabel('z2')
ax2.grid(True)

"""
for x,y in zip(z1,z2):
    label = "(" + "{:.1f}".format(x) + "," + "{:.1f}".format(y) + ")"
    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,-5)) # distance from text to points (x,y)
"""
plt.savefig('pareto_compromise_z.png')

# Write the results of the Pareto Frontier to a file
with open('results_compromise.txt',mode='w') as results_file:
    results_file.write('Variable x\n\n')
    results_file.write('   x1 = ' + str(x1) + '\n\n')
    results_file.write('   x2 = ' + str(x2) + '\n\n')
    results_file.write('   z1 = ' + str(z1) + '\n\n')
    results_file.write('   z2 = ' + str(z2) + '\n')

res = {'x1':x1,'x2':x2,'x1':x1,'z2':z2}
df = pandas.DataFrame(res, columns=["x1","x2","z1","z2"])
df.to_csv(r'results.csv', index=False, header=True)

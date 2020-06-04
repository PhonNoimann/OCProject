# P2: Bi-Objective linear optimization problem Constraint method

import csv
import pandas
import os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import *
from functions import feasPlot

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

for eps in range(31):
    for j in range(len(obj_file)):
        #model.c.add(expr = (None, sum(model.x[i]*obj_file[i][j] for i in A), c_file['UB'][2]))
        model.c.add(expr = (None, sum(model.x[i]*obj_file[i][j] for i in A), eps))
    # Define the objectives
    model.obj_1 = Objective(
        expr = sum(model.x[i]*obj_file[i][0]for i in A), sense = minimize)
    model.obj_2 = Objective(
        expr = sum(model.x[i]*obj_file[i][1]for i in A), sense = minimize)

    model.obj_1.activate()
    model.obj_2.deactivate()
    model.c[4].deactivate()
    opt = SolverFactory('glpk')
    results = opt.solve(model)
    #results.write()
    #model.solutions.load_from(results)
    #model.pprint()
    x1 = np.array(value(model.x['x1']))
    x2 = np.array(value(model.x['x2']))
    z1 = np.array(value(model.x['x1'])*value(obj_file['x1'][0])+value(model.x['x2'])*value(obj_file['x2'][0]))
    z2 = np.array(value(model.x['x1'])*value(obj_file['x1'][1])+value(model.x['x2'])*value(obj_file['x2'][1]))

    model.obj_2.activate()
    model.obj_1.deactivate()
    model.c[4].deactivate()
    model.c[5].activate()
    results = opt.solve(model)
    #results.write()
    #model.solutions.load_from(results)

    x1 = np.append(x1,value(model.x['x1']))
    x2 = np.append(x2,value(model.x['x2']))
    z1 = np.append(z1,value(model.x['x1'])*value(obj_file['x1'][0])+value(model.x['x2'])*value(obj_file['x2'][0]))
    z2 = np.append(z2,value(model.x['x1'])*value(obj_file['x1'][1])+value(model.x['x2'])*value(obj_file['x2'][1]))
    model.del_component(model.c[4])


model.pprint()
print(z1)
print(z2)
print(x1)
print(x2)

fig2, ax2 = plt.subplots()
ax2.plot(z1,z2,'.r')
ax2.plot(z1,z2,'b')
ax2.set_title('Pareto frontier Constraint method z\'s values')
ax2.set_xlabel('z1')
ax2.set_ylabel('z2')
ax2.grid(True)
for x,y in zip(z1,z2):

    label = "(" + "{:.1f}".format(x) + "," + "{:.1f}".format(y) + ")"

    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,-5)) # distance from text to points (x,y)


plt.savefig('pareto_constraint_z.png')
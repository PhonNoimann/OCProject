import csv
import pandas
import os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import *
#from functions import *

# Check if data for the objective functions and constraints exists, if not it creates them
files = os.listdir()
if ('constraints.csv' in files)==False or ('objectives.csv' in files)==False:
    with open("dataCreator.py", mode='r') as dataCreator:
        exec(dataCreator.read())

# Read the values of the objective functions and constraints
obj_file = pandas.read_csv('objectives.csv')
c_file = pandas.read_csv('constraints.csv')
A = obj_file.columns.tolist()
B = sum(1 for line in obj_file)
print(obj_file)
print(c_file)
print(A)
print(B)
x1 = []
x2 = []
z1 = []
z2 = []
z3 = []
mu1 = []
mu2 = []
mu3 = []

# Create the model
model = ConcreteModel()
model.x = Var( A, within = NonNegativeReals)
model.c = ConstraintList()
opt = SolverFactory('glpk')
for j in range(len(c_file)):
    model.c.add(expr = (None, sum(model.x[i]*c_file[i][j] for i in A), c_file['UB'][j]))
# Define the objectives

model.obj_1 = Objective(
    expr = sum(model.x[i]*obj_file[i][0]for i in A), sense = minimize)
model.obj_2 = Objective(
    expr = sum(model.x[i]*obj_file[i][1]for i in A), sense = minimize)
model.obj_3 = Objective(
    expr = sum(model.x[i]*obj_file[i][2]for i in A), sense = minimize)

model.obj_1.activate()
model.obj_2.deactivate()
model.obj_3.deactivate()
results1 = opt.solve(model)

mu1g = value(model.x['x1'])*value(obj_file['x1'][0])+value(model.x['x2'])*value(obj_file['x2'][0])
mu2.append(value(model.x['x1'])*value(obj_file['x1'][1])+value(model.x['x2'])*value(obj_file['x2'][1]))
mu3.append(value(model.x['x1'])*value(obj_file['x1'][2])+value(model.x['x2'])*value(obj_file['x2'][2]))

model.obj_2.activate()
model.obj_1.deactivate()
results2 = opt.solve(model)

mu1.append(value(model.x['x1'])*value(obj_file['x1'][0])+value(model.x['x2'])*value(obj_file['x2'][0]))
mu2g = value(model.x['x1'])*value(obj_file['x1'][1])+value(model.x['x2'])*value(obj_file['x2'][1])
mu3.append(value(model.x['x1'])*value(obj_file['x1'][2])+value(model.x['x2'])*value(obj_file['x2'][2]))

model.obj_3.activate()
model.obj_2.deactivate()
results3 = opt.solve(model)

mu1.append(value(model.x['x1'])*value(obj_file['x1'][0])+value(model.x['x2'])*value(obj_file['x2'][0]))
mu2.append(value(model.x['x1'])*value(obj_file['x1'][1])+value(model.x['x2'])*value(obj_file['x2'][1]))
mu3g = value(model.x['x1'])*value(obj_file['x1'][2])+value(model.x['x2'])*value(obj_file['x2'][2])

mu1b = max(mu1)
mu2b = max(mu2)
mu3b = max(mu3)

print('mu1g = ' + str(mu1g))
print('mu1b = ' + str(mu1b))
print('mu2g = ' + str(mu2g))
print('mu2b = ' + str(mu2b))
print('mu3g = ' + str(mu3g))
print('mu3b = ' + str(mu3b))

model.obj_3.deactivate()

W1 = []
for r in range(6):
    W1.append(r/5)
W2 = W1.copy()

for w1 in W1:
    for w2 in W2:
        w3 = 1 - w1 - w2
        w3 = round(w3,2)
        w = [w1,w2,w3]
        model.obj = Objective(
        expr = (obj_file['x1'][0]*model.x['x1']+obj_file['x2'][0]*model.x['x2']-mu1g)/(mu1b-mu1g)*w[0] + (obj_file['x1'][1]*model.x['x1']+obj_file['x2'][1]*model.x['x2'])/(mu2b-mu2g)*w[1] + (obj_file['x1'][2]*model.x['x1']+obj_file['x2'][2]*model.x['x2'])/(mu3b-mu3g)*w[2], sense = minimize)
    
        opt = SolverFactory('glpk')
        results = opt.solve(model)
        x1.append(value(model.x['x1']))
        x2.append(value(model.x['x2']))
        z1.append(value(model.x['x1'])*value(obj_file['x1'][0])+value(model.x['x2'])*value(obj_file['x2'][0]))
        z2.append(value(model.x['x1'])*value(obj_file['x1'][1])+value(model.x['x2'])*value(obj_file['x2'][1]))
        z3.append(value(model.x['x1'])*value(obj_file['x1'][2])+value(model.x['x2'])*value(obj_file['x2'][2]))
        model.del_component(model.obj)
    del W2[-1]

print('x1 = ' + str(x1))
print('x2 = ' + str(x2))
print('z1 = ' + str(z1))
print('z2 = ' + str(z2))
print('z3 = ' + str(z3))

"""
c_arr = np.array(c_file)
obj_arr = np.array(obj_file)
Plots(c_arr, obj_arr, x1, x2, z1, z2, 'goal programming')
"""

# Write the results of the Pareto Frontier to a file
res = {'x1':x1,'x2':x2,'z1':z1,'z2':z2,'z3':z3}
df = pandas.DataFrame(res, columns=["x1","x2","z1","z2"])
df.to_csv(r'results_goal.csv', index=False, header=True)

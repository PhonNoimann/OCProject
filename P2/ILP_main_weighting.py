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

# Create the model
model = ConcreteModel()
model.x = Var( A, within = NonNegativeIntegers)
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
obj_arr = np.array(obj_file)
Plots(c_arr, obj_arr, x1, x2, z1, z2, 'ILP weighting')

# Write the results of the Pareto Frontier to a file
res = {'x1':x1,'x2':x2,'z1':z1,'z2':z2}
df = pandas.DataFrame(res, columns=["x1","x2","z1","z2"])
df.to_csv(r'ILP_results_weighting.csv', index=False, header=True)
    

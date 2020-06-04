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
for j in range(len(obj_file)):
    model.c.add(expr = (None, sum(model.x[i]*obj_file[i][j] for i in A), c_file['UB'][2]))
# Define the objectives
model.obj_1 = Objective(
    expr = sum(model.x[i]*obj_file[i][0]for i in A), sense = minimize)
model.obj_2 = Objective(
    expr = sum(model.x[i]*obj_file[i][1]for i in A), sense = minimize)

model.obj_1.activate()
model.obj_2.deactivate()
model.c[4].deactivate()
opt = SolverFactory('glpk')
results = opt.solve(model, tee=True)
results.write()
model.solutions.load_from(results)
model.pprint()
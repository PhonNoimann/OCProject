#main2.py
import csv
import pandas
import os
import fnmatch
from pyomo.environ import *

# Check if data for the objective functions and constraints exist, if not create them
files = os.listdir()
if ('constraints.csv' in files)==False or ('objectives.csv' in files)==False:
    with open("dataCreator.py", mode='r') as dataCreator:
        exec(dataCreator.read())

obj_file = pandas.read_csv('objectives.csv')
c_file = pandas.read_csv('constraints.csv')
A = obj_file.columns.tolist()


# Creates a new model
model = ConcreteModel()

# Variable initialization
model.x = Var( A, within = NonNegativeReals)

# Objective function

model.obj = Objective(
    expr = sum(model.x[i]*obj_file[i][0]for i in A), sense = minimize)

# Constraints
model.c = ConstraintList()
c_file = pandas.read_csv('constraints.csv')
for j in range(len(c_file)):
    model.c.add(expr = (None, sum(model.x[i]*c_file[i][j] for i in A), c_file['UB'][j]))

# Creo il solver e risolvo il problema
opt = SolverFactory('glpk')
results = opt.solve(model, tee=True)
results.write()
model.solutions.load_from(results)

with open('results.txt',mode='w') as results_file:
    for v in model.component_objects(Var, active=True):
        results_file.write('Variable ' + str(v) + '\n')
        varobject = getattr(model, str(v))
        for index in varobject:
            results_file.write('   '+str(index) + ' = ' + str(varobject[index].value) + '\n')
    


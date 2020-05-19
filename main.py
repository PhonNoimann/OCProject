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

# Creates a new model
model = ConcreteModel()

# Variable initialization
model.x1 = Var(within = NonNegativeReals)
model.x2 = Var(within = NonNegativeReals)

# Objective function
obj_file = pandas.read_csv('objectives.csv')
model.obj = Objective(
    expr = model.x1*obj_file['x1'][0] + model.x2*obj_file['x2'][0], sense = minimize)

# Constraints
model.c = ConstraintList()
c_file = pandas.read_csv('constraints.csv')
for i in range(len(c_file)):
    model.c.add(expr = (None, model.x1*c_file['x1'][i] + model.x2*c_file['x2'][i], c_file['UB'][i]))

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
            results_file.write('   '+str(index) + ' ' + str(varobject[index].value) + '\n')
    

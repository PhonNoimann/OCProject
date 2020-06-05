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
B = sum(1 for line in obj_file)
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
# Define the objectives

 model.obj_1 = Objective(
    expr = sum(model.x[i]*obj_file[i][0]for i in A), sense = minimize)
 model.obj_2 = Objective(
    expr = sum(model.x[i]*obj_file[i][1]for i in A), sense = minimize)

model.obj_1.activate()
model.obj_2.deactivate()

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
            

model.obj_2.activate()
model.obj_1.deactivate()

opt2 = SolverFactory('glpk')
results2 = opt2.solve(model, tee=True)
results2.write()
model.solutions.load_from(results2)
with open('results2.txt',mode='w') as results2_file:
    for v in model.component_objects(Var, active=True):
        results2_file.write('Variable ' + str(v) + '\n')
        varobject2 = getattr(model, str(v))
        for index2 in varobject2:
            results2_file.write('   '+str(index2) + ' = ' + str(varobject2[index2].value) + '\n')
            
file = open('results.txt', 'r')
variabili = file.readlines()
x=[None]*len(variabili)
for i in range(1,len(variabili)):
      x[i] = float(variabili[i][8:])
x.remove(None)
file2=open('objectives.csv','r')
coeff=file2.readlines()
A1=[None]*len(coeff)
for j in range(0,len(coeff),2):
    A1[j]=float(coeff[2][j])
A1.remove(None)
for k in range (len(A1)):
    C=sum(x[k]*A1[k])
print(C)





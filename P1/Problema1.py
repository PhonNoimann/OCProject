# Problema1.py
from pyomo.environ import *

# Creo il modello
model = ConcreteModel()

# Inizializzo le variabili
model.x1 = Var(within = NonNegativeReals)
model.x2 = Var(within = NonNegativeReals)

# Imposto la funzione obiettivo
model.obj = Objective(
    expr = -3*model.x1 - 8*model.x2, sense = minimize)

# Imposto i vincoli
model.c1 = Constraint(
    expr = 2*model.x1 + 6*model.x2 <= 27)
model.c2 = Constraint(
    expr = 3*model.x1 + 2*model.x2 <= 16)
model.c3 = Constraint(
    expr = 4*model.x1 + model.x2 <= 18)

# Creo il solver e risolvo il problema
opt = SolverFactory('glpk')
result_obj = opt.solve(model, tee = True)

model.pprint()
#modifica
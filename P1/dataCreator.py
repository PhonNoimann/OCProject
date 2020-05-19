# data files creator

import csv

# objective function
with open('objectives.csv', mode='w') as obj_file:
    fieldnames = ['x1','x2']
    obj_writer = csv.DictWriter(obj_file, fieldnames=fieldnames, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)

    obj_writer.writeheader()
    obj_writer.writerow({'x1':-3,'x2':-8})

# constraints
with open('constraints.csv', mode='w') as c_file:
    fieldnames = ['x1','x2','LB','UB']
    c_writer = csv.DictWriter(c_file, fieldnames=fieldnames, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)

    c_writer.writeheader()
    c_writer.writerow({'x1':2,'x2':6,'LB':None,'UB':27})
    c_writer.writerow({'x1':3,'x2':2,'LB':None,'UB':16})
    c_writer.writerow({'x1':4,'x2':1,'LB':None,'UB':18})

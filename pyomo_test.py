from __future__ import division
from pyomo.environ import *
from pyomo.opt import SolverFactory
import numpy as np
model = ConcreteModel()

model.A = Set(initialize=['x1','x2'])
lb = {'x1':-150, 'x2':-150}
ub = {'x1':150, 'x2':150}

def fb(model, i):
  return (lb[i], ub[i])

model.x = Var(model.A, within=Reals, bounds=fb)

def objRule(model):
  return 0.585*np.exp(127)*model.x['x1'] + 0.255*np.exp(125)*model.x['x2']**2 - 0.542*np.exp(123)*model.x['x1']**3 - 0.223*np.exp(123)*model.x['x2']**3
model.obj = Objective(rule=objRule, sense=maximize)

opt = SolverFactory('baron')
opt.solve(model)
model.display()

for ele in model.x:
  print(value(model.x[ele]))
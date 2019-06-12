from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
from SALib.util import read_param_file
import numpy as np

def transpose(matrix):
        new_matrix = []
        for i in range(len(matrix[0])):
            matrix1 = []
            for j in range(len(matrix)):
                matrix1.append(matrix[j][i])
            new_matrix.append(matrix1)
        return new_matrix

problem = {
  'num_vars': 3,
  'names': ['x1', 'x2', 'x3'],
  'bounds': [[-np.pi, np.pi]]*3
}

# Generate samples
param_values = saltelli.sample(problem, 1000)

# Run model (example)
Y = Ishigami.evaluate(param_values)

# Perform analysis
Si = sobol.analyze(problem, Y, print_to_console=True)
# Returns a dictionary with keys 'S1', 'S1_conf', 'ST', and 'ST_conf'
# (first and total-order indices with bootstrap confidence intervals)
problem2 = read_param_file('SALib_bound.txt')
param_values2 = saltelli.sample(problem2, 20)
# print(param_values.shape)
# print(Y)
# print(Si)
for i,x in enumerate(param_values2):
  print(x)
a = [[-np.pi, np.pi]]*3
print(transpose(a))
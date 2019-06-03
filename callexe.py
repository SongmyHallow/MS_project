import os
import sys
import re
import matplotlib.pyplot as plt
import random
import numpy as np
from SALib.sample import saltelli
from SALib.util import read_param_file

# compile .c file
def compileSource(file):
    os.system('gcc source_princetonlibgloballib/'+file+'.c -lm -o '+file)

# helper function to generate input value
def val_generate(lb,ub,numofvar):
    bound_lst = []
    name_lst = []
    index = 1
    for l, u in zip(lb,ub):
        bound_lst.append([l,u])
        name_lst.append('x'+str(index))
        index = index+1
    problem = {
        'num_vars': numofvar,
        'names': name_lst,
        'bounds': bound_lst
    }
    param_values = saltelli.sample(problem,20)
    np.savetxt("param_values.txt", param_values)
    return param_values


# generate input.in file according to requested number of input values
# output: input.in file
def create_input(file,values):
    infile = open(file, 'w')
    for val in values:
        infile.write(str(val)+'\n')
    infile.close()

# read returned value from file
# input: input file, name of list
def read_output(file,lst,compilefile):
    readfile = open(file, 'r')
    # outfile = open(file,'w')
    for line in readfile.readlines():
        lst.append(float(line.strip()))
        # outfile.write(line)
    readfile.close()
    # outfile.close()

def read_input(file,lst,compilefile):
    temp = []
    readfile = open(file,'r')
    for line in readfile.readlines():
        temp.append(float(line.strip()))
    readfile.close()
    lst.append(temp)

# call the executable file repeatedly and generate the plot
# input: input file name, number of variables, number of loops
# output: returned values list
def repeat_call(infile,compilefile,outfile,lb,ub,loop,numofvar):
    input_values = val_generate(lb,ub,numofvar)
    outlst = []
    inlst = []
    for i in range(len(input_values)):
        create_input(infile,input_values[i])
        os.system('./'+compilefile)
        read_output(outfile,outlst,compilefile)
        read_input(infile,inlst,compilefile)
    return outlst,inlst

# read data file to get values of number of vars, boundaries nad starting points
# input: data file name
# output: attributes
def read_datafile(file):
    numOfVar = 0
    lowBound = []
    upBound = []
    startPoint = []

    infile = open(file,'r')
    lines = infile.readlines()
    # The first line
    for num in lines[0].split():
        numOfVar = int(num.strip())
    # The second line
    for i in lines[1].split():
        lowBound.append(float(i.strip()))
    # The third line
    for j in lines[2].split():
        upBound.append(float(j.strip()))
    # The fourth line
    for k in lines[3].split():
        startPoint.append(float(k.strip()))
    infile.close()
    return numOfVar,lowBound,upBound,startPoint

#***************************************************************
# The following is the main function
# command line: python .py c_code_filename input.in output.out
#***************************************************************

if __name__ == '__main__':
    compileFile = sys.argv[1]
    dataFile = sys.argv[2]
    inputFile = sys.argv[3]
    outputFile = sys.argv[4]

def main():
    # os.system("mkdir outfiles/"+compileFile)
    compileSource(compileFile)
    numOfVar, lb, ub, sp = read_datafile(dataFile)
    print(numOfVar, lb, ub, sp)
    ydata,in_values = repeat_call(inputFile, compileFile, outputFile, lb, ub, 10, numOfVar)
    print(ydata,in_values)



main()



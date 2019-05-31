import os
import sys
import re
import matplotlib.pyplot as plt
import random
import numpy as np

# compile .c file
def compileSource(file):
    os.system('gcc source_princetonlibgloballib/'+file+'.c -lm -o outfiles/'+file+'/'+file)

# helper function to generate input value
# input: index of line
# output: (string) value
def val_generate(boundary):
    lb, ub = boundary
    val = random.uniform(lb,ub)
    return str(val)

# generate input.in file according to requested number of input values
# output: input.in file
def create_input(file,lb,ub,compilefile):
    infile = open('outfiles/'+compilefile+'/'+file, 'w')
    for i in zip(lb,ub):
        infile.write(val_generate(i)+"\n")
    infile.close()

# read returned value from file
# input: input file, name of list
def read_output(file,lst,compilefile):
    readfile = open(file, 'r')
    outfile = open('outfiles/'+compilefile+'/'+file,'w')
    for line in readfile.readlines():
        lst.append(float(line.strip()))
        outfile.write(line)
    readfile.close()
    outfile.close()

def read_input(file,lst,compilefile):
    temp = []
    readfile = open('outfiles/'+compilefile+'/'+file,'r')
    for line in readfile.readlines():
        temp.append(float(line.strip()))
    readfile.close()
    lst.append(temp)

# call the executable file repeatedly and generate the plot
# input: input file name, number of variables, number of loops
# output: returned values list
def repeat_call(infile,compilefile,outfile,lb,ub,loop):
    outlst = []
    inlst = []
    for i in range(loop):
        create_input(infile,lb,ub,compilefile)
        os.system('./outfiles/'+compilefile+'/'+compilefile)
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
    ydata,in_values = repeat_call(inputFile, compileFile, outputFile, lb, ub, 10)
    print(ydata,in_values)


main()
# testFile = open('DataFileName.txt','r')
# for line in testFile.readlines():
#     print(line)


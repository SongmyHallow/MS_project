import os
import sys
import re
import matplotlib.pyplot as plt
import random
import numpy as np

if __name__ == '__main__':
    compileFile = sys.argv[1]
    inputFile = sys.argv[2]
    outputFile = sys.argv[3]
    dataFile = sys.argv[4]

# read .c file and match the line of for loop to get the number of variables
# input: name of input file
# output: (integer) number of variables
def get_number(file):
    numOfVar = -1
    infile = open(file+'.c','r')
    for line in infile.readlines():
        matchObj = re.match(r'for',line,re.M|re.I)
        if matchObj:
            line_group1 = line.split(';')
            line_group2 = line_group1[1].split('<')
            numOfVar = line_group2[1].strip()
    return int(numOfVar)

# helper function to generate input value
# input: index of line
# output: (string) value
def val_generate(index,loop):
    return str(0.01/index*loop)

# generate input.in file according to requested number of input values
# output: input.in file
def create_input(file,number,loop):
    infile = open(file, 'w')
    for i in range(number):
        infile.write(val_generate((i+1),loop)+"\n")
    infile.close()

# read returned value from file
# input: input file, name of list
def read_output(file,lst):
    readfile = open(file, 'r')
    for line in readfile.readlines():
        lst.append(float(line.strip()))
    readfile.close()

def read_input(file,lst):
    temp = []
    readfile = open(file,'r')
    for line in readfile.readlines():
        temp.append(float(line.strip()))
    readfile.close()
    lst.append(temp)

# call the executable file repeatedly and generate the plot
# input: input file name, number of variables, number of loops
# output: returned values list
def repeat_call(infile,compilefile,outfile,number,loop):
    outlst = []
    inlst = []
    for i in range(loop):
        create_input(infile, number, i)
        os.system('.\\'+compilefile)
        read_output(outfile,outlst)
        read_input(infile,inlst)
    # print(outlst)
    return outlst,inlst

def read_datafile(file):
    numOfVar = []
    lowBound = []
    upBound = []
    startPoint = []

    infile = open(file,'r')
    lines = infile.readlines()
    # The first line
    for num in lines[0].split():
        numOfVar.append(int(num.strip()))
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
    print(numOfVar,lowBound,upBound,startPoint)
    return numOfVar,lowBound,upBound,startPoint

#***************************************************************
# The following is the main function
# command line: python .py c_code_filename input.in output.out
#***************************************************************

# get the number of variables
numOfVar = get_number(compileFile)
# compile .c file
os.system('gcc '+compileFile+'.c -o '+compileFile)

# Use loop to repeatedly call executable file
# infile = open(inputFile, 'a')
ydata,input_values = repeat_call(inputFile,compileFile,outputFile,numOfVar,1000)
xdata = [i for i in range(len(ydata))]
# plt.plot(xdata,ydata)
# plt.show()

read_datafile(dataFile)
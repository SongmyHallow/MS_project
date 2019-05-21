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

# read .c file and match the line of for loop to get
# the number of variables
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

def read_output(file,lst):
    readfile = open(file, 'r')
    for line in readfile.readlines():
        lst.append(float(line.strip()))
    readfile.close()

# get the number of variables
numOfVar = get_number(compileFile)
# compile .c file
os.system('gcc '+compileFile+'.c -o '+compileFile)

# Use loop to repeatedly call executable file
# infile = open(inputFile, 'a')
outlst = []
for i in range(1000):
    create_input(inputFile, numOfVar, i)
    os.system('.\\'+compileFile)
    read_output(outputFile, outlst)
print(outlst)

xdata = [i for i in range(len(outlst))]
plt.plot(xdata,outlst)
plt.show()


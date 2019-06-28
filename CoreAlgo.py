# Compile .c file
# input: name of c file (with out extension)
import os
def compileCode(filename):
    os.system('gcc source_princetonlibgloballib/'+filename+'.c -lm -o '+filename)

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
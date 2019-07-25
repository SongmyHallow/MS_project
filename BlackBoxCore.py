'''
MS project algorithm
Muyi Song (muyis)
Latest update: 7/25/2019

command line syntax: python BlackBox.py (name of model file) (number of iterations)
                                            sys.argv[1]             sys.argv[2]
sampling methods:
1. Hammersley sequence
2. Ver der Corput sequence
3. Halton sequence
4. Latin Random

search algorithm:
1. Coordinate search (shuffle the order of variables every iteration)
=================================================================================
'''
import os
import sys

"""
================================================================================
support functions
================================================================================
"""
def writeInput(filename,input_values):
    infile = open(filename, 'w')
    for val in input_values:
        infile.write(str(val)+'\n')
    infile.close()

def readOutput(filename):
    readfile = open(filename, 'r')
    line = readfile.readline()
    if(line.strip() == "1.#INF00000000000"):
        return "INF"
    else:
        output_value = float(line.strip())
        return output_value
    
def genBlackBoxValue(filename,input_values):
    input_filename = "input.in"
    output_filename = "output.out"
    
    writeInput(input_filename,input_values)
    os.system('.\\'+filename)
    output_value = readOutput(output_filename)
    return output_value

class blackBox(object):
    radius = 0.5
    def __init__(self,name=None,cycles=None,radius=None,
                 numOfVar=None,
                 lowBound=[],upBound=[],
                 iniStart=[],backUpStart=[],actualStart=[]):
        self.name = name
        self.cycles = cycles
        self.radius = radius
        self.numOfVar = numOfVar
        self.lowBound = lowBound
        self.upBound = upBound
        self.iniStart = iniStart
        self.backUpStart = backUpStart
        self.actualStart = actualStart

    def clear(self):
        self.name=None
        self.cycles=None
        self.radius=None
        self.numOfVar = None
        self.lowBound = []
        self.upBound = []
        self.iniStart = []
        self.backUpStart = []
        self.actualStart = []
    
    def showParameter(self):
        print("Name:",self.name)
        print("Total number of cycles:",self.cycles)
        print("Search radius:",self.radius)
        print("Number of variables:",self.numOfVar)
        print("lowBound:",self.lowBound)
        print("upBound:",self.upBound)
        print("Initial starting point from data file:",self.iniStart)
        print("Back up starting points:",self.backUpStart)
        print("Actual starting point is:",self.actualStart)
        
    '''
    Compile .c file
    :param string filename: name of c file (with out extension)
    ''' 
    def compileCode(self):
        os.system('gcc source_princetonlibgloballib/'+self.name+'.c -lm -o '+self.name)
        after_name = self.name+".exe"
        print("Blackbox Model Name: ",self.name)
        if(os.path.exists(after_name)):
            print("Compilation finished")
        else:
            print("Compilation failed")
    
    
    '''
    Read boundaries, starting points and number of variables
    :param string filename: name of c file (with out extension)
    '''
    def readDataFile(self):
        infile = open("problemdata/"+self.name+".problem.data",'r')
        lines = infile.readlines()
        # The first line
        for num in lines[0].split():
            self.numOfVar = int(num.strip())
        # The second line
        for i in lines[1].split():
            self.lowBound.append(float(i.strip()))
        # The third line
        for j in lines[2].split():
            self.upBound.append(float(j.strip()))
        # The fourth line
        for k in lines[3].split():
            self.iniStart.append(float(k.strip()))
        infile.close()
    '''
    Generate starting points that are ready to be chose
    '''
    def genBackupStart(self):
        import numpy as np    
        # TODO: Finally I want to generate 10 backup starting points, which can be modifed later
        for i in range(len(self.lowBound)):
            arr = np.linspace(self.lowBound[i],self.upBound[0],12)
            self.backUpStart.append(arr)
        self.backUpStart = np.transpose(self.backUpStart)[1:-1]
    
    '''
    Choose a starting point from backup and check if it is valid,
    otherwise remove that point from list and choose another one
    '''
    def genActualStart(self):
        import random
        self.actualStart = random.choice(self.backUpStart)
        output = genBlackBoxValue(self.name,self.actualStart)
        while(output == "INF"):
            self.backUpStart.remove(self.actualStart)
            self.actualStart = random.choice(self.backUpStart)
            output = genBlackBoxValue(self.name,self.actualStart)
    
    

    def genVariableBound(self,index):
        if(self.actualStart[index]-self.radius<self.lowBound[index]):
            lb = self.lowBound[index]
        else:
            lb = self.actualStart[index]-self.radius
        
        if(self.actualStart[index]+self.radius>self.upBound[index]):
            ub = self.upBound[index]
        else:
            ub = self.actualStart[index]+self.radius
        return lb,ub
        
    
    def coordinateSearch(self):
        import random
        from Sampling import halton_sequence,hammersley_sequence,van_der_corput,latin_random_sequence
        for cycle in range(self.cycles):
            print("The No.",cycle+1,"Cycle")
            shuffleOrder = list(range(self.numOfVar))
            random.shuffle(shuffleOrder)
            for indexOfVar in shuffleOrder:
                lb,ub = self.genVariableBound(indexOfVar)
                print(indexOfVar,shuffleOrder)


"""
================================================================================
main function
================================================================================
"""
if __name__ == "__main__":
    fileName = sys.argv[1]
    cycles = int(sys.argv[2])

def main():
    import time
    box = blackBox(name=fileName,cycles=cycles,radius=1.0)
    box.compileCode()
    box.readDataFile()
    box.genBackupStart()
    box.genActualStart()
    box.coordinateSearch()
    box.showParameter()

main()
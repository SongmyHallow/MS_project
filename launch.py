import os
import re

ModelFile = open('DataFileName.txt','r')

for line in ModelFile.readlines():
    if(line.strip() =='arith.h' or line.strip() == 'config.h'):
        continue
    else:
        os.system("python CoreAlgo.py "+line.strip()+" 1")
    
ModelFile.close()
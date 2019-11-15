import os
import re

ModelFile = open('selected.txt','r')

for line in ModelFile.readlines():
    # if(line.strip() =='arith.h' or line.strip() == 'config.h' or line.strip() == 'expquad'):
    #     continue
    # else:
    #     os.system("python BlackBoxCore.py "+line.strip()+" 60")
    print(line.strip())
    
ModelFile.close()
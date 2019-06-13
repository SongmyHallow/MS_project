#!/bin/zsh
localFile=$"/mnt/c/Users/tjuso/Documents/Graduate/MS_project/cam6.lst"
if [ -f $testFile ];then
    sudo rm -i "$testFile"
else
    echo "File does not exist"
fi

rm 'logscratch'
rm 'cam6.exe'
rm 'almopt.txt'
rm 'trace.trc'
rm 'simwrapper.py'
rm 'cam6cv.py'
rm 'z1.py'
rm 'output.out'
rm 'input.in'
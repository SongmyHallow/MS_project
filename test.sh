#!/bin/bash
testFile=$"/mnt/c/Users/tjuso/Documents/Graduate/MS_project/DataFileName.txt"
if [ -f $testFile ];then
    sudo rm -i "$testFile"
else
    echo "File does not exist"
fi

# read names of data files from /problemdata folder
# for 480s
# path=$"/mnt/c/Users/tjuso/Documents/Graduate/MS_project/problemdata" 
# for surface
path=$"/mnt/c/MYSong/Graduate/MS_project/problemdata"
files=$(ls $path)
for fullname in $files
do
    echo ${fullname%.*.*} >> DataFileName.txt
done


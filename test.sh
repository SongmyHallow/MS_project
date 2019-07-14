#!/bin/bash
testFile=$"/mnt/c/Users/tjuso/Documents/Graduate/MS_project/DataFileName.txt"
# if [ -f $testFile ];then
#     sudo rm -i "$testFile"
# else
#     echo "File does not exist"
# fi

# read names of data files from /problemdata folder
# for 480s
path=$"/mnt/c/Users/tjuso/Documents/Graduate/MS_project/source_princetonlibgloballib" 
# for surface
# path=$"/mnt/c/MYSong/Graduate/MS_project/problemdata"
files=$(ls $path)
for fullname in $files
do
    echo ${fullname%.*c*} >> DataFileName.txt
done

# cat $testFile | while read line
# do
#     echo $line
#     python3 CoreAlgo.py $line 1
# done


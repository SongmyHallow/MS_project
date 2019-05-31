#!/bin/bash
testFile=$"/mnt/c/Users/tjuso/Documents/Graduate/MS_project/DataFileName.txt"
if [ -f $testFile ];then
    sudo rm -i "$testFile"
else
    echo "File does not exist"
    # exit 1
fi

# read names of data files from /problemdata folder
path=$"/mnt/c/Users/tjuso/Documents/Graduate/MS_project/problemdata"
files=$(ls $path)
for fullname in $files
do
    echo ${fullname%.*.*} >> DataFileName.txt
done

file=$1
file_run=`awk -v str=${file} 'BEGIN{len=split(str,str_list,".");print str_list[1]}'`
gcc -o ${file_run} $file   
[ $? -eq 0 ] && ./${file_run}  

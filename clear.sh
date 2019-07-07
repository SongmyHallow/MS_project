#!/bin/zsh
#surface version
DATE=$(date -d '-1 month' +%Y%m%d)
File1="/mnt/c/Users/tjuso/Documents/Graduate/MS_project/z1.py"
File2="/mnt/c/Users/tjuso/Documents/Graduate/MS_project/almopt.txt"
File3="/mnt/c/Users/tjuso/Documents/Graduate/MS_project/simwrapper.py"
File4="/mnt/c/Users/tjuso/Documents/Graduate/MS_project/logscratch"
File5="/mnt/c/Users/tjuso/Documents/Graduate/MS_project/cam6"
File6="/mnt/c/Users/tjuso/Documents/Graduate/MS_project/input.in"
File7="/mnt/c/Users/tjuso/Documents/Graduate/MS_project/output.out"


if [ -f "$File1" ];then
    sudo rm -f "$File1"
    echo $File1" is deleted"
else
    echo $File1" does not exist"
fi

if [ -f "$File2" ];then
    sudo rm -f "$File2"
    echo $File2" is deleted"
else
    echo $File2" does not exist"
fi

if [ -f "$File3" ];then
    sudo rm -f "$File3"
    echo $File3" is deleted"
else
    echo $File3" does not exist"
fi

if [ -f "$File4" ];then
    sudo rm -f "$File4"
    echo $File4" is deleted"
else
    echo $File4" does not exist"
fi

if [ -f "$File5" ];then
    sudo rm -f "$File5"
    echo $File5" is deleted"
else
    echo $File5" does not exist"
fi

if [ -f "$File6" ];then
    sudo rm -f "$File6"
    echo $File6" is deleted"
else
    echo $File6" does not exist"
fi
if [ -f "$File7" ];then
    sudo rm -f "$File7"
    echo $File7" is deleted"
else
    echo $File7" does not exist"
fi

find . -name "*.exe" | xargs rm -rf
echo "Executable files are deleted"

#!/bin/bash

# read names of data files from /problemdata folder
path=$"/mnt/c/Users/tjuso/Documents/Graduate/MS_project/problemdata"
files=$(ls $path)
for filename in $files
do
    echo $filename >> DataFileName.txt
done

# read names of source code files from /source_princetonlibgloballib
path=$"/mnt/c/Users/tjuso/Documents/Graduate/MS_project/source_princetonlibgloballib"
files=$(ls $path)
for filename in $files
do
    echo $filename >> SourceCodeName.txt
done

#!/bin/bash

echo -n "Please enter a foldername: "
read foldername
path=./Plots/$foldername
echo $path
mkdir $path

mv *png $path
mv output.txt $path
cp config.yaml $path

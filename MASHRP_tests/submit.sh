#!/bin/bash

cd runs

for file in run_*
do

cd $file

if [ ! -f ziying.out ]; then

echo $file
sbatch job_run

fi

cd ..

done

#!/bin/bash

cd runs

for file in run_*
do

echo $file

cd $file

cp ../../job_run .
cp ../../run_dynamics.py .

sed -i "s/name/$file/g" job_run

run=`echo $file | awk -F_ '{print 4500*$2+95500}'`
mvrun=`echo $file | awk -F_ '{print 8200+1800*$2}'`

sed -i "s/XXX/$run/g" run_dynamics.py
sed -i "s/YYY/$mvrun/g" run_dynamics.py

cd ..

done


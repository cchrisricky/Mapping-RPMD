#!/bin/bash

> didnt_finish.dat

for file in runs/run_*
do

echo $file

if [ ! -f $file/ziying.out ]; then
    echo $file | sed -e 's\runs/\\g' >> didnt_finish.dat
else
    chk=`tail -1 $file/ziying.out | head -1`
    if [ "$chk" != "done" ]; then
        echo $file | sed -e 's\runs/\\g' >> didnt_finish.dat

        ########DOUBLE CHECK THIS!!!!!####### 
        #cd $file
        #cp ../../remove.sh .
        #./remove.sh
        #rm remove.sh
        #cd ../../
        #####################################

    fi
fi

done


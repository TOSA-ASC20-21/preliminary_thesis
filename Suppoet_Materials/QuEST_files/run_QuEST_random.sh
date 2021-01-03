#!/bin/bash

QUEST=/home/john/Others/Playground/QuEST-master

cd $QUEST
cp benchmark/* ./
cmake -DUSER_SOURCE="random.c" -DGPUACCELERATED=0 -DDISTRIBUTED=0
make
(time ./demo) > log.QuEST_random 2>&1
rm ./demo

exit 0

#!/bin/bash

QUEST=/home/john/Others/Playground/QuEST-master

cd $QUEST
cp benchmark/* ./
cmake -DUSER_SOURCE="GHZ_QFT.c" -DGPUACCELERATED=0 -DDISTRIBUTED=0
make
(time ./demo) > log.QuEST_GHZ_QFT 2>&1
rm ./demo

exit 0

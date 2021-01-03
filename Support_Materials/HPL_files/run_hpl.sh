#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries/linux/lib/intel64_lin/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64/openmpi/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/john/hpl/src/cuda

mpirun -np 2 -hostfile nodes ./xhpl


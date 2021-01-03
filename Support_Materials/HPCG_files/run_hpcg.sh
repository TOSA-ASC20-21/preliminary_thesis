#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/john/anaconda3/pkgs/cudatoolkit-10.0.130-0/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64/openmpi/lib/
export CUDA_VISIBLE_DEVICES=0,1
export PATH=$PATH:/usr/lib64/openmpi/bin
#./xhpcg-3.1_gcc_485_cuda-10.0.130_ompi-3.1.0_sm_35_sm_50_sm_60_sm_70_sm_75_ver_10_9_18
mpirun -np 2 ./xhpcg-3.1_gcc_485_cuda-10.0.130_ompi-3.1.0_sm_35_sm_50_sm_60_sm_70_sm_75_ver_10_9_18 | tee ./results/xhpcg_gpu.log

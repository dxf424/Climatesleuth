#!/bin/bash -l 

#PBS -N training
#PBS -l walltime=00:20:00
#PBS -l select=1:ncpus=8:ngpus=4:mem=128GB 
#PBS -A NAML0001
#PBS -q develop 
#main 

### Merge output and error files
#PBS -j oe
#PBS -k eod

source ~/.bashrc
module load conda
conda activate hwtmode
cd /glade/work/fanda/github/miles-climatesleuth/ 
pip install .
cd scripts
torchrun --nproc_per_node=4 train_VGG_LWP.py 

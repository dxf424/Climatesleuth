#!/bin/bash -l 

#PBS -N evaluation_analysis
#PBS -l walltime=00:35:00
#PBS -l select=1:ncpus=8:ngpus=1:mem=128GB 
#PBS -A NAML0001
#PBS -q casper 
#switch to casper if your walltime exceeds 30 min

### Merge output and error files
#PBS -j oe
#PBS -k eod

source ~/.bashrc
module load conda
conda activate hwtmode

# Set multithreading for libraries
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTORCH_NUM_THREADS=8

# change it to your own directory
cd /glade/work/fanda/github/miles-climatesleuth/scripts
# change it to your own directory
torchrun --nproc_per_node=1 --directory /glade/work/fanda/miles-climatesleuth/microphysics_analysis
# change it to your own directory
torchrun --nproc_per_node=1 --directory /glade/work/fanda/miles-climatesleuth/microphysics_analysis

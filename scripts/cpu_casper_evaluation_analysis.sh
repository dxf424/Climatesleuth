#!/bin/bash -l 

#PBS -N evaluation_analysis
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=8:mem=128GB 
#PBS -A NAML0001
#PBS -q casper 
#switch to casper if your walltime exceeds 30 min

### Merge output and error files
#PBS -j oe
#PBS -k eod

source ~/.bashrc
module load conda
conda activate hwtmode
cd /glade/work/fanda/github/miles-climatesleuth/scripts
python evaluation_VGG_LWP.py 
python analysis_VGG_LWP.py

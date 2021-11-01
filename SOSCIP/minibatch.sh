#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=12:00:00
#SBATCH --job-name=generate_XRD_job
#SBATCH --output=/scratch/k/kleiman/mccrinbc/data/new_sims/logfiles/simlog_%j.txt
#SBATCH --mail-type=FAIL

cd $SCRATCH

module load NiaEnv/2019b python/3.8 
module load NiaEnv/2019b gnu-parallel 
source ~/.virtualenvs/XRD/bin/activate 

parallel --joblog parallel.log -j 10 "python $HOME/Create_XRD_Spectra_minibatch.py --data_path $SCRATCH/data/material_params_v2/batch{}* --save_path $SCRATCH/data/new_sims/ --index_begin $1 --batch_size $2 --chunk_size $3" ::: {0..9}

cd $HOME
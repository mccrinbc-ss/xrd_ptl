#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=12:00:00
#SBATCH --job-name=create_batches
#SBATCH --output=/scratch/k/kleiman/mccrinbc/data/agg_data_2/simlog_%j.txt
#SBATCH --mail-type=FAIL

cd $SCRATCH

module load NiaEnv/2019b python/3.8 
module load NiaEnv/2019b gnu-parallel 
source ~/.virtualenvs/XRD/bin/activate 

parallel --joblog parallel.log -j 10 "python $HOME/chunks_to_batches.py --data_dir $SCRATCH/data/new_sims --save_dir $SCRATCH/data/agg_data_2 --batch_num {}" ::: {0..9}

cd $HOME
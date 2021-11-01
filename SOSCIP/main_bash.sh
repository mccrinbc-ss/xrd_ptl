#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=12:00:00
#SBATCH --job-name=generate_XRD_job
#SBATCH --output=/scratch/k/kleiman/mccrinbc/data/sim_results/logfiles/simlog_%j.txt
#SBATCH --mail-type=FAIL

MAX_SIZE=150000
CHUNK_SIZE=100
BATCH_SIZE=1000

for ((INDEX_BEGIN=36600;INDEX_BEGIN<=$MAX_SIZE;INDEX_BEGIN+=$BATCH_SIZE)); do
	. minibatch.sh $INDEX_BEGIN $BATCH_SIZE $CHUNK_SIZE 
done

#FIRST ITERATION 
# These are where the sets stopped each time. If we have problems with the total number of files,
# it's probably because we need to run a subset of dataframes. 
# 36600
# 73800
# 110500
# 147400

# 149100
# 94700 #for the last batch of data


#SECOND ITERATION: 
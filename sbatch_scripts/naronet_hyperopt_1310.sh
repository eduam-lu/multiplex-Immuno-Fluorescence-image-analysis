#!/bin/bash
# 13/10
#
# Time
#SBATCH -t 100:00:00
#
# job name
#SBATCH -J NaroNet_final_hyperopt_clusters
#
# GPU, CPU and memory request
#
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH -p gpua100i
#
# output filenames stdout and stderr - customise, include %j
#SBATCH -o sbatch_outputs/NaroNet_hyperopt_2010_%j.out
#SBATCH -e sbatch_outputs/NaroNet_hyperopt_2010_%j.out

#SBATCH --mail-user=edu.amgo@gmail.com
#SBATCH --mail-type=ALL

# Description
# PCL model already trained. Using actual images and labels
# I start the final architecture search

# Now I am exploring the clusters using the fixed architecture

# write this script to stdout-file - useful for scripting errors
cat $0

# load the modules required for you program - customise for your program
module load GCC/10.2.0
module load CUDA/11.1.1
module load cuDNN/8.0.4.30-CUDA-11.1.1
module load Anaconda3
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
conda activate NaroNet

# MAIN EXECUTION
# customise for your program name and add arguments if required
cd /home/eduamgo/NaroNet/
python /home/eduamgo/NaroNet/main.py
#!/bin/bash
# 08/09
#
# Time
#SBATCH -t 02:10:00
#
# job name
#SBATCH -J NaroNet_POLEx
#
# GPU, CPU and memory request
#
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p gpua40
#
# output filenames stdout and stderr - customise, include %j
#SBATCH -o NaroNet_POLEx_%j.out
#SBATCH -e NaroNet_POLEx_%j.out

#SBATCH --mail-user=edu.amgo@gmail.com
#SBATCH --mail-type=ALL


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
python /home/eduamgo/NaroNet/main.py
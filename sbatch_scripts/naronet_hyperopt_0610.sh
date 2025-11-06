#!/bin/bash
# 06/10
#
# Time
#SBATCH -t 30:00:00
#
# job name
#SBATCH -J NaroNet_full_fixed_params
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
#SBATCH -o NaroNet_full1_0610_%j.out
#SBATCH -e NaroNet_full1_0610_%j.out

#SBATCH --mail-user=edu.amgo@gmail.com
#SBATCH --mail-type=ALL

# Description
# PCL model already trained. Using actual images and labels
# Run a full model with the default params as ray is being tricky

# I do 50 epochs with 10 fold validation, default parameters, no hyperopt.

# Write this script to stdout-file - useful for scripting errors
cat $0

# load the modules required for you program - customise for your program
module load GCC/10.2.0
module load CUDA/11.1.1
module load cuDNN/8.0.4.30-CUDA-11.1.1
module load Anaconda3
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
conda activate NaroNet


echo $CUDA_VISIBLE_DEVICES
nvidia-smi
# MAIN EXECUTION
# customise for your program name and add arguments if required
python /home/eduamgo/NaroNet/main.py
#!/bin/bash
# 03/10
#
# Time
#SBATCH -t 24:00:00
#
# job name
#SBATCH -J NaroNet_hyperopt_trial_1_rerun
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
#SBATCH -o NaroNet_hyperopt1_0310_%j.out
#SBATCH -e NaroNet_hyperopt1_0310_%j.out

#SBATCH --mail-user=edu.amgo@gmail.com
#SBATCH --mail-type=ALL

# Description
# PCL model already trained. Using actual images and labels
# First hyperparameter optimization run. This sbatch testes:
# 25 architectures, 8 fold CV, with ASHA scheduler max 40 epochs and reduction factor 3

# Hyperopt search was successful but ray stopped. I relaunch this to move forward from the architecture search
# This is still TRIAL 1! Same parameters

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
#!/bin/bash
# 07/10
#
# Time
#SBATCH -t 20:00:00
#
# job name
#SBATCH -J NaroNet_bioinsights_0810
#
# GPU, CPU and memory request
#
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH -p gpua100
#
# output filenames stdout and stderr - customise, include %j
#SBATCH -o sbatch_outputs/NaroNet_bioinsights_0810_%j.out
#SBATCH -e sbatch_outputs/NaroNet_bioinsights_0810_%j.out

#SBATCH --mail-user=edu.amgo@gmail.com
#SBATCH --mail-type=ALL

# Description
# PCL model already trained. Using actual images and labels
# Run a full model with the default params as ray is being tricky

# I trained Naronet with 50 epochs, 5 fold and a lower learning rate 0.0001, which ended up working
# Bioinsights bugged, I am running this with a clean image representation folder!

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
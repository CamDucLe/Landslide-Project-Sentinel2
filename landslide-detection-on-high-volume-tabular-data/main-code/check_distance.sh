#!/bin/bash

#SBATCH --job-name=landslide-tabular
#SBATCH --ntasks=1
#SBATCH --nodelist=s3ls2002
####SBATCH --gres=gpu:titan:1
####SBATCH --gres=gpu:2080:1
source /opt/anaconda3/etc/profile.d/conda.sh

module load use.storage
module load anaconda3

#conda activate jupyter_hpc2 #HPC1
conda activate jupyter_2  #HPC2 

export HDF5_USE_FILE_LOCKING=FALSE
export PATH="/usr/lib/x86_64-linux-gnu/:$PATH"


##---
## python3 statistic.py
python3 euclid_space.py



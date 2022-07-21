#!/bin/bash -l
#SBATCH --gpus=1
#SBATCH -p long
#SBATCH -o err_cheetah.out

source .dmcbash
conda activate rad
export MUJOCO_GL = "egl"
cd /nethome/dyung6/share4_dyung6/CURL2
python train.py --config_file cheetah.json

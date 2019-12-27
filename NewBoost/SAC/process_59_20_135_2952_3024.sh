#!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6  # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=00-14:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --error=%N-%j.err   # %N for node name, %j for jobID
#SBATCH --job-name=SAC_59_20_135_2952_3024

module load python/3.6.3
module load scipy-stack
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
pip install torch --no-index
pip install tqdm

python main.py 0.2 13.5 5 ../data/added_hhdata_59_2.csv 2952 3024
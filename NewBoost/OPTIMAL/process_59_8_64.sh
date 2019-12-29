#!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6  # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=00-00:10      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --error=%N-%j.err   # %N for node name, %j for jobID
#SBATCH --job-name=OPT59_8_64

module load python/3.6.3
module load scipy-stack
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate

python main.py -0.08 6.4 2 ../data/added_hhdata_59_2.csv
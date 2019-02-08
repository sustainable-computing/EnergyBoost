#!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6  # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=00-14:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --error=%N-%j.err   # %N for node name, %j for jobID
#SBATCH --job-name=solarRL

### UNCOMMENT THE NEXT TWO LINES IF RUNNING ON COMPUTE CANADA INFRASTRUCTURE
module load cuda cudnn python/3.5.2
source tensorflow/bin/activate


for i in 2002
do
  #python test_ddpg.py 0.2 0 0 data_predicted2/predicted_hhdata_"$i"_2.csv
  python3 solar_a2c_linear.py 0.2 6.4 2 data_predicted2/predicted_hhdata_"$i"_2.csv
  python3 solar_a2c_linear.py 0.2 13.5 5 data_predicted2/predicted_hhdata_"$i"_2.csv
  #python test_ddpg.py 0.04 0 0 data_predicted2/predicted_hhdata_"$i"_2.csv
  python3 solar_a2c_linear.py 0.04 6.4 2 data_predicted2/predicted_hhdata_"$i"_2.csv
  python3 solar_a2c_linear.py 0.04 13.5 5 data_predicted2/predicted_hhdata_"$i"_2.csv
  #python test_ddpg.py 0.08 0 0 data_predicted2/predicted_hhdata_"$i"_2.csv
  python3 solar_a2c_linear.py 0.08 6.4 2 data_predicted2/predicted_hhdata_"$i"_2.csv
  python3 solar_a2c_linear.py 0.08 13.5 5 data_predicted2/predicted_hhdata_"$i"_2.csv
  #python test_ddpg.py 0.1 0 0 data_predicted2/predicted_hhdata_"$i"_2.csv
  python3 solar_a2c_linear.py 0.1 6.4 2 data_predicted2/predicted_hhdata_"$i"_2.csv
  python3 solar_a2c_linear.py 0.1 13.5 5 data_predicted2/predicted_hhdata_"$i"_2.csv
done

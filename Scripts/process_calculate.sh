#!/bin/bash
#python generatepower.py
#for i in 26 59 77 86 93 101 114 171 1086 1403
for i in 871
do
  #python test_ddpg.py 0.2 0 0 data_filled2/processed_hhdata_"$i"_2.csv
  python calculate_solver_bill_new.py 0.2 6.4 2 data_filled2/processed_hhdata_"$i"_2.csv
  python calculate_solver_bill_new.py 0.2 13.5 5 data_filled2/processed_hhdata_"$i"_2.csv
  #python test_ddpg.py 0.04 0 0 data_filled2/processed_hhdata_"$i"_2.csv
  python calculate_solver_bill_new.py 0.04 6.4 2 data_filled2/processed_hhdata_"$i"_2.csv
  python calculate_solver_bill_new.py 0.04 13.5 5 data_filled2/processed_hhdata_"$i"_2.csv
  #python test_ddpg.py 0.08 0 0 data_filled2/processed_hhdata_"$i"_2.csv
  python calculate_solver_bill_new.py 0.08 6.4 2 data_filled2/processed_hhdata_"$i"_2.csv
  python calculate_solver_bill_new.py 0.08 13.5 5 data_filled2/processed_hhdata_"$i"_2.csv
  #python test_ddpg.py 0.1 0 0 data_filled2/processed_hhdata_"$i"_2.csv
  python calculate_solver_bill_new.py 0.1 6.4 2 data_filled2/processed_hhdata_"$i"_2.csv
  python calculate_solver_bill_new.py 0.1 13.5 5 data_filled2/processed_hhdata_"$i"_2.csv
done

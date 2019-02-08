#!/bin/bash
#python generatepower.py
for i in "/home/baihong/Documents/result/data_filled4"/*
do
  #python test_ddpg.py 0.2 0 0 "$i"
  python calculate_solver_bill_new.py 0.2 6.4 2 "$i"
  python calculate_solver_bill_new.py 0.2 13.5 5 "$i"
  #python test_ddpg.py 0.04 0 0 "$i"
  python calculate_solver_bill_new.py 0.04 6.4 2 "$i"
  python calculate_solver_bill_new.py 0.04 13.5 5 "$i"
  #python test_ddpg.py 0.08 0 0 "$i"
  python calculate_solver_bill_new.py 0.08 6.4 2 "$i"
  python calculate_solver_bill_new.py 0.08 13.5 5 "$i"
  #python test_ddpg.py 0.1 0 0 "$i"
  python calculate_solver_bill_new.py 0.1 6.4 2 "$i"
  python calculate_solver_bill_new.py 0.1 13.5 5 "$i"
done

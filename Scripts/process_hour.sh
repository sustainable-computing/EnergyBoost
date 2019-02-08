#!/bin/bash
#python generatepower.py

for i in 2002
do
    echo "$i"
    python MPC_hour2.py -0.04 6.4 2 data_added2/added_hhdata_"$i"_2.csv
    python MPC_hour2.py -0.04 13.5 5 data_added2/added_hhdata_"$i"_2.csv
    python MPC_hour.py -0.04 6.4 2 data_added2/added_hhdata_"$i"_2.csv
    python MPC_hour.py -0.04 13.5 5 data_added2/added_hhdata_"$i"_2.csv
done

#!/bin/bash

for i in 26 59 77 86 93 94 101 114 171 187;
    do

      python3 main.py 0.2 6.4 2 ../data/added_hhdata_"$i"_2.csv
      python3 main.py 0.2 13.5 5 ../data/added_hhdata_"$i"_2.csv
      
      python3 main.py 0.04 6.4 2 ../data/added_hhdata_"$i"_2.csv
      python3 main.py 0.04 13.5 5 ../data/added_hhdata_"$i"_2.csv
      
      python3 main.py 0.08 6.4 2 ../data/added_hhdata_"$i"_2.csv
      python3 main.py 0.08 13.5 5 ../data/added_hhdata_"$i"_2.csv
      
      python3 main.py 0.1 6.4 2 ../data/added_hhdata_"$i"_2.csv
      python3 main.py 0.1 13.5 5 ../data/added_hhdata_"$i"_2.csv
    done

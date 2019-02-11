# EnergyBoost: Learning-based Control of Home Batteries

## Description
EnergyBoost employs accurate supervised learning models 
for predicting the next day available solar energy and household demand, and 
physical models that closely approximate the output of a solar inverter
and the state of charge (SoC) of a lithium-ion battery. 
It formulates the control of battery charge and discharge operations 
as an optimal control problem over a finite time horizon.

## Prerequisites
EnergyBoost is python-based and you can easily install all required library through
`pip install` or `conda install` for conda users. 

Please refer to [Gurobi webpage](http://www.gurobi.com/index) to install
the Gurobi Programming solver for optimization.

## Table of contents
### [Data prepocessing](https://github.com/sustainable-computing/EnergyBoost/tree/master/data_processing)
* [Raw data processing](https://github.com/sustainable-computing/EnergyBoost/blob/master/data_processing/raw_data_processing.ipynb) merges home load data together with associated solar data and seprated them by home id. Output files are saved as `dataprocessed_hhdata_<home id>.csv`
* 

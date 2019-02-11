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
* [Raw data processing](https://github.com/sustainable-computing/EnergyBoost/blob/master/data_processing/raw_data_processing.ipynb) merges home load data together with associated solar data and seprated them by home id. Output files are saved as `data_by_home/processed_hhdata_<home id>.csv`

* [Fill missing data](https://github.com/sustainable-computing/EnergyBoost/blob/master/data_processing/fill_missing_data.ipynb)
will fill any missing data for a year by filling the gap by last available value. 

* [Generate ac power](https://github.com/sustainable-computing/EnergyBoost/blob/master/data_processing/generatepower.py)This is a converter uses pvlib to calulate the ac power of solar output, it will add add one more new column to the data file including the power values.

* [Add history](https://github.com/sustainable-computing/EnergyBoost/blob/master/data_processing/add_history.rb)will match data for each time slot with its history data of previous time slot and data one week before, these data can be used as features for prediction.

* [GHI model figure](https://github.com/sustainable-computing/EnergyBoost/blob/master/data_processing/ghi_model_fig.ipynb) and [HL model figure](https://github.com/sustainable-computing/EnergyBoost/blob/master/data_processing/hl_model_fig.ipynb) compare differnt models for predicting ac power and home load and generate a plot showing the nRMSE of different models.

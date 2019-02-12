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

### Required Packages and versions

| Packages      | Version       |
| ------------- |:-------------:| 
| pandas        | 0.23.0 |
| numpy         |   1.14.2   |
| pvlib         |   0.5.2    |
| matplotlib    | 2.1.2 |
| scikit-learn  | 0.20.2 |
| tensorflow-gpu | 1.12.0 |
| cxvpy         | 1.0.6 |
| gurobipy      | 8.1.0 |
| gym           | 0.10.5 |







Please refer to [Gurobi webpage](http://www.gurobi.com/index) to install
the Gurobi Programming solver for optimization.

## Table of contents
### [Data prepocessing](https://github.com/sustainable-computing/EnergyBoost/tree/master/data_processing)
* [Raw data processing](https://github.com/sustainable-computing/EnergyBoost/blob/master/data_processing/raw_data_processing.py) merges home load data together with associated solar data and seprated them by home id. Output files are saved as `data_by_home/processed_hhdata_<home id>.csv`

* [Fill missing data](https://github.com/sustainable-computing/EnergyBoost/blob/master/data_processing/fill_missing_data.py)
will fill any missing data for a year by filling the gap by last available value. 

* [Generate ac power](https://github.com/sustainable-computing/EnergyBoost/blob/master/data_processing/generatepower.py)This is a converter uses pvlib to calulate the ac power of solar output, it will add add one more new column to the data file including the power values.

* [Add history](https://github.com/sustainable-computing/EnergyBoost/blob/master/data_processing/add_history.rb)will match data for each time slot with its history data of previous time slot and data one week before, these data can be used as features for prediction.

* [GHI model figure](https://github.com/sustainable-computing/EnergyBoost/blob/master/data_processing/ghi_model_fig.py) and [HL model figure](https://github.com/sustainable-computing/EnergyBoost/blob/master/data_processing/hl_model_fig-more.py) compare differnt models for predicting ac power and home load and generate a plot showing the nRMSE of different models.


* [GHI model](https://github.com/sustainable-computing/EnergyBoost/blob/master/data_processing/ghi_model.py) and [HL model](https://github.com/sustainable-computing/EnergyBoost/blob/master/data_processing/hl_model.py) train the best models for all homes predicting ac output and homeload. The trainted models are saved so that it can be used repeatly without training again.

* [Get preidct data](https://github.com/sustainable-computing/EnergyBoost/blob/master/data_processing/get_predict_data.py)call the saved models from last step and saved pridicted home load and ac output for each home. It merges the predcit two values with other features of states and save the output as `predicted_data/predicted_hhdata_<home id>.csv`.

### [Baseline](https://github.com/sustainable-computing/EnergyBoost/tree/master/Baseline)
Contains different rule based baselines for different scenarios.
* [Get no solar](https://github.com/sustainable-computing/EnergyBoost/blob/master/Baseline/get_nosolar.py) Get baseline strategy for no solar and no batery installed.

* [Get no battery](https://github.com/sustainable-computing/EnergyBoost/blob/master/Baseline/get_nostorage.py) Get baseline strategy for no battery installed and with solar installed.

* [Get RBC](https://github.com/sustainable-computing/EnergyBoost/blob/master/Baseline/get_rbc.py) Get a baseline strategy based on rule based TOU price. 

These three implementations will call assocaited environments developed for these different scenarios.


### [DLC](https://github.com/sustainable-computing/EnergyBoost/tree/master/DLC)
Contains implementation of direct mapping method.
* [Mapping](https://github.com/sustainable-computing/EnergyBoost/blob/master/DLC/mapping.py) Mapping from avaiable features of each time slots to optimal actions solved by the solver and oracle data. 



### [MPC](https://github.com/sustainable-computing/EnergyBoost/tree/master/MPC)
This part contains implementation of model predictive control. 
* [The main code](https://github.com/sustainable-computing/EnergyBoost/blob/master/MPC/MPC.py) for MPC uses cvxpy and Gurobi solver to solve the problem as a convex optimization problem, several versions of the MPC code are used for solve problem in different scenarios like hourly price and Net metering.

* [Create hourly price](https://github.com/sustainable-computing/EnergyBoost/blob/master/MPC/create_hourly_price_table.py) changes unix time to UTC time and generate a hourly price table

* [Create TOU price table](https://github.com/sustainable-computing/EnergyBoost/blob/master/MPC/create_tou_price.py) creates a TOU price table for a year, so it can be used loaded as matrix in the solver.

### [RL](https://github.com/sustainable-computing/EnergyBoost/tree/master/RL)
Implementation of Reinforcement Learning method
* [A2C](https://github.com/sustainable-computing/EnergyBoost/blob/master/RL/solar_a2c_nonlinear.py) Implementation of Actor Critic method which will genetrate the stratgy given states and reward.

* [Environment](https://github.com/sustainable-computing/EnergyBoost/blob/master/RL/environment.py) Formulate the problem into MDP environment, contains features of reset initialize states, calulate reward of a given states, and simulate next states.

### [Compile Results and Plots](https://github.com/sustainable-computing/EnergyBoost/tree/master/Plot)
Scripts for compile results and genrate plots:
* [Get Feasible Actions](https://github.com/sustainable-computing/EnergyBoost/blob/master/Plot/calculate_solver_bill_new.py) Results returned by the contoller using predicted data are sometimes not feasible, it will map the learned action to feasible actions and recalculate the bills and record the feasible actions.

* [Compile results of all homes](https://github.com/sustainable-computing/EnergyBoost/blob/master/Plot/compile_bill_sce.py) will compile results of all controllers of one given home or all homes and requried scenrios, will write results of the recalated bill in to a new table and create line plots of cumulative bills of a year. 

* [Plot ground truth](https://github.com/sustainable-computing/EnergyBoost/blob/master/Plot/PolicyViz.ipynb) will plot one week ground truth of one home including home load, solar output and optimal action.

* [Plot controller action](https://github.com/sustainable-computing/EnergyBoost/blob/master/Plot/PolicyViz-controller.ipynb) will plot learned policy of all controllers of one specific period of year. 

* [Genrate ROI table](https://github.com/sustainable-computing/EnergyBoost/blob/master/Plot/bill_table2.py) will read in bill results of all controllers and calulate needed information for ROI.

* [Calulate PAR of contoller](https://github.com/sustainable-computing/EnergyBoost/blob/master/Plot/mpc_par.ipynb) will calulate the average PAR of all homes of all days of a year.

* [Generate violin plots of bills of all controllers](https://github.com/sustainable-computing/EnergyBoost/blob/master/Plot/violinplot_price4.py)will create violin plots of all contollers to show the distrubution of bills of all homes of a given scenario.

### Usage and shell scripts
For code of all controllers: 

`python <controller_method>.py <solar tariff price> <battery size> <Maximum Charging/Discharging rate> <Data file of one home>`
  
e.g. `python MPC.py -0.2 6.4 2 processed_hhdata_59_2.csv`

#### [Some tools for running scripts](https://github.com/sustainable-computing/EnergyBoost/tree/master/Scripts)
* [Run all scenarios given home id](https://github.com/sustainable-computing/EnergyBoost/blob/master/Scripts/process_calculate.sh)will run all scenarios of homes given home id.

* [Run all scenarios give filder name](https://github.com/sustainable-computing/EnergyBoost/blob/master/Scripts/process_calculate_batch.sh)will run all scenarios of all homes inside one given folder.

* [Run scripts on Compute Canada solver](https://github.com/sustainable-computing/EnergyBoost/blob/master/Scripts/process_base.sh) when running simnulations on compute canada server like Graham, hearder file for assign compute resources for one job need to be added. 

For more details about schedule jobs on compute canada, please refer to [running jobs on Compute Canada](https://docs.computecanada.ca/wiki/Running_jobs)

* [List of all home id](https://github.com/sustainable-computing/EnergyBoost/blob/master/Scripts/all_id.py) generate list of all home id which will be used to create seprate jobs for different homes on server.

* [Create shell scripts for one given home](https://github.com/sustainable-computing/EnergyBoost/blob/master/Scripts/create-scipts.py)will create shell scripts to run scenarios for all given home on server.

* [Run indiviual jobs for all homes](https://github.com/sustainable-computing/EnergyBoost/blob/master/Scripts/run_all_scripts.sh) will create one single jobs for all homes so that mutiple jobs can be run simultaneously on Compute Canada server.
































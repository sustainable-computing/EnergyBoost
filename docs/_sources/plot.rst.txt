**************
Plot
**************

Scripts for compile results and genrate plots:

Bill
===============
* `Get Feasible Actions <https://github.com/sustainable-computing/EnergyBoost/blob/master/Plot/calculate_solver_bill_new.py>`_ Results returned by the contoller using predicted data are sometimes not feasible, it will map the learned action to feasible actions and recalculate the bills and record the feasible actions.

* `Compile results of all homes <https://github.com/sustainable-computing/EnergyBoost/blob/master/Plot/compile_bill_sce.py>`_ will compile results of all controllers of one given home or all homes and requried scenrios, will write results of the recalated bill in to a new table and create line plots of cumulative bills of a year. 

Policy
==========
* `Plot ground truth <https://github.com/sustainable-computing/EnergyBoost/blob/master/Plot/PolicyViz.ipynb>`_ will plot one week ground truth of one home including home load, solar output and optimal action.

* `Plot controller action <https://github.com/sustainable-computing/EnergyBoost/blob/master/Plot/PolicyViz-controller.ipynb>`_ will plot learned policy of all controllers of one specific period of year. 

Violin
============
* `Genrate ROI table <https://github.com/sustainable-computing/EnergyBoost/blob/master/Plot/bill_table2.py>`_ will read in bill results of all controllers and calulate needed information for ROI.

* `Calulate PAR of contoller <https://github.com/sustainable-computing/EnergyBoost/blob/master/Plot/mpc_par.ipynb>`_ will calulate the average PAR of all homes of all days of a year.

* `Generate violin plots of bills of all controllers <https://github.com/sustainable-computing/EnergyBoost/blob/master/Plot/violinplot_price4.py>`_ will create violin plots of all contollers to show the distrubution of bills of all homes of a given scenario.

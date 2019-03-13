***********************
Methods documentation
***********************


.. toctree::
   :maxdepth: 3


MPC
===========
This part contains implementation of model predictive control.

Main code
------------
`The main code <https://github.com/sustainable-computing/EnergyBoost/blob/master/MPC/MPC2.py>`_ for MPC uses cvxpy and Gurobi solver to solve the problem as a convex optimization problem, several versions of the MPC code are used for solve problem in different scenarios like hourly price and Net metering.

.. automodule:: MPC.MPC2
    :members:
    :undoc-members:
    :show-inheritance:

Create hourly price
-----------------------
`Create hourly price <https://github.com/sustainable-computing/EnergyBoost/blob/master/MPC/create_hourly_price_table.py>`_ changes unix time to UTC time and generate a hourly price table

.. automodule:: MPC.create_hourly_price_table
    :members:
    :undoc-members:
    :show-inheritance:

Create TOU price
---------------------
`Create TOU price table <https://github.com/sustainable-computing/EnergyBoost/blob/master/MPC/create_tou_price.py>`_ creates a TOU price table for a year, so it can be used loaded as matrix in the solver.


RL
===========
Implementation of Reinforcement Learning method

A2C
------

`A2C <https://github.com/sustainable-computing/EnergyBoost/blob/master/RL/solar_a2c_nonlinear.py>`_ Implementation of Actor Critic method which will genetrate the stratgy given states and reward.

.. automodule:: RL.solar_a2c_nonlinear
    :members:
    :undoc-members:
    :show-inheritance:

Environment
---------------
`Environment <https://github.com/sustainable-computing/EnergyBoost/blob/master/RL/environment.py>`_ Formulate the problem into MDP environment, contains features of reset initialize states, calulate reward of a given states, and simulate next states.

.. automodule:: RL.environment
    :members:
    :undoc-members:
    :show-inheritance:

Baseline
==============
Contains different rule based baselines for different scenarios.

* `Get no solar <https://github.com/sustainable-computing/EnergyBoost/blob/master/Baseline/get_nosolar.py>`_ Get baseline strategy for no solar and no batery installed.

* `Get no battery <https://github.com/sustainable-computing/EnergyBoost/blob/master/Baseline/get_nostorage.py>`_ Get baseline strategy for no battery installed and with solar installed.

* `Get RBC <https://github.com/sustainable-computing/EnergyBoost/blob/master/Baseline/get_rbc.py>`_ Get a baseline strategy based on rule based TOU price. 

These three implementations will call assocaited environments developed for these different scenarios.


DLC
==========
Contains implementation of direct mapping method.

* `Mapping <https://github.com/sustainable-computing/EnergyBoost/blob/master/DLC/mapping.py>`_ Mapping from avaiable features of each time slots to optimal actions solved by the solver and oracle data. 

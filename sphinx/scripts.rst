********************************
Some tools for running scripts
********************************
 .. toctree::
   :maxdepth: 2


Run local batch
==================
* `Run all scenarios given home id <https://github.com/sustainable-computing/EnergyBoost/blob/master/Scripts/process_calculate.sh>`_ will run all scenarios of homes given home id.

* `Run all scenarios give filder name <https://github.com/sustainable-computing/EnergyBoost/blob/master/Scripts/process_calculate_batch.sh>`_ will run all scenarios of all homes inside one given folder.


Run on Compute Canada jobs
====================================================
* `Run scripts on Compute Canada solver <https://github.com/sustainable-computing/EnergyBoost/blob/master/Scripts/process_base.sh>`_ when running simnulations on compute canada server like Graham, hearder file for assign compute resources for one job need to be added.

For more details about schedule jobs on compute canada, please refer to [running jobs on Compute Canada <https://docs.computecanada.ca/wiki/Running_jobs>`_

* `List of all home id <https://github.com/sustainable-computing/EnergyBoost/blob/master/Scripts/all_id.py>`_ generate list of all home id which will be used to create seprate jobs for different homes on server.

* `Create shell scripts for one given home <https://github.com/sustainable-computing/EnergyBoost/blob/master/Scripts/create-scipts.py>`_ will create shell scripts to run scenarios for all given home on server.

* `Run indiviual jobs for all homes <https://github.com/sustainable-computing/EnergyBoost/blob/master/Scripts/run_all_scripts.sh>`_ will create one single jobs for all homes so that mutiple jobs can be run simultaneously on Compute Canada server.


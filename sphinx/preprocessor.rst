************************
Data prepossessing
************************
 .. toctree::
   :maxdepth: 2

Raw Data processing
========================
* `Raw data processing <https://github.com/sustainable-computing/EnergyBoost/blob/master/data_processing/raw_data_processing.py>`_ merges home load data together with associated solar data and seprated them by home id. Output files are saved as `data_by_home/processed_hhdata_<home id>.csv`


* `Fill missing data <https://github.com/sustainable-computing/EnergyBoost/blob/master/data_processing/fill_missing_data.py>`_ will fill any missing data for a year by filling the gap by last available value.

Physical Model
======================
* `Generate ac power <https://github.com/sustainable-computing/EnergyBoost/blob/master/data_processing/generatepower.py>`_ is a converter uses pvlib to calulate the ac power of solar output, it will add add one more new column to the data file including the power values.


* `Add history <https://github.com/sustainable-computing/EnergyBoost/blob/master/data_processing/add_history.rb>`_ will match data for each time slot with its history data of previous time slot and data one week before, these data can be used as features for prediction.

Data Driven Model
===================
* `GHI model figure <https://github.com/sustainable-computing/EnergyBoost/blob/master/data_processing/ghi_model_fig.py>`_ and `HL model figure <https://github.com/sustainable-computing/EnergyBoost/blob/master/data_processing/hl_model_fig-more.py>`_ compare differnt models for predicting ac power and home load and generate a plot showing the nRMSE of different models.


* `GHI model <https://github.com/sustainable-computing/EnergyBoost/blob/master/data_processing/ghi_model.py>`_ and `HL model <https://github.com/sustainable-computing/EnergyBoost/blob/master/data_processing/hl_model.py>`_ train the best models for all homes predicting ac output and homeload. The trainted models are saved so that it can be used repeatly without training again.

* `Get preidct data <https://github.com/sustainable-computing/EnergyBoost/blob/master/data_processing/get_predict_data.py>`_ call the saved models from last step and saved pridicted home load and ac output for each home. It merges the predcit two values with other features of states and save the output as `predicted_data/predicted_hhdata_<home id>.csv`.
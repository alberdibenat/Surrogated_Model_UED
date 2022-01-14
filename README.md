# Surrogated_Model_UED
Surrogated model of the UED version of the SRF Photoinjector based on neural networks.  
Includes:  
  1. Surrogated_model.ipynb contains the surrogated model, the data points are taken from the already filtered In/Out.txt.
  2. Filter_data.ipynb takes the raw data in X/Y_values_SC_FBL_Updated.txt and does a pre-exploration to filter out unwanted data points. It computes the time of flight jitter corresponding to each data point and saves the results in In/Out.txt
  3. ToF_Jitter_callable.py contains the class definition to compute the time of flight jitter. It is called by Filter_data.ipynb.
  4. X/Y_values_SC_FBL_updated.txt includes the X/Y raw data points.
  5. Results_inverted_0323.txt is a text file needed to calculate the time of flight jitter. It is called by ToF_Jitter_callable.py.
  6. In/Out.txt contains the already filtered data points used for training.
  6. model.h5 contains the trained surrogated model for UED.

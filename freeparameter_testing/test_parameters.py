
# %% Imports

# Imports from the library
from micorr.simulation import simulations, testing, plotting, transformations
from micorr.estimators import mi_estimators

# More general imports
import numpy as np
import matplotlib.pyplot as plt

# %% Initial parameters

# Simulating the base evoked response
sim_params = {'start': -0.05, 'end': 0.3,
              'peak_time': 0.1, 'peak_duration': 0.12,
              'sin_shift': 0.01, 'fs': 600}
evoked, times = simulations.simulate_evoked(**sim_params)

# transformed version of the base signal
_, qua_signal = transformations.transf_power(evoked)

# Number of noise iterations to use
n_runs = 100

# SNR of the added noise
snr = 20

#%% Impact of the choice of k

est_test = 'AB' # Defining the MI estimator to test (Options: KSG, KDE and AB)

if est_test == 'KSG':
    est_func  = mi_estimators.mi_ksg
    param_name = 'k' 
    param_values = [1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60] # ks to test (can be changed)

if est_test == 'KDE':
    est_func  = mi_estimators.kde_mi
    param_name = 'bw_method' 
    param_values = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2] # Bandwidth factors to test (can be changed)
    
if est_test == 'AB':
    est_func  = mi_estimators.binned_mi
    param_name = 'n_bins' 
    param_values = [1, 2, 3, 5, 10, 13, 15, 20, 25, 27, 30]   # Number of bins to test (can be changed)

# testing the impact of the parameter choice with two identical responses
kres_linear = testing.test_est_params(comp1=evoked,    # Signal 1 to compare
                                      comp2=evoked,    # Signal 2 to compare
                                      est=est_func,     # Estimator function to test
                                      param_name=param_name,  # Function parameter to change
                                      param_values=param_values, # Parameter values to test
                                      snr=snr,         # Noise level (dB) to use
                                      n_runs=n_runs    # Number of noise iterations to use
                                      )

# Testing the impact of the parameter choice when comparing with noise
kres_lower = testing.test_est_params(comp1=evoked, 
                                     comp2=np.std(evoked), # Standard deviation of the noise signal to compare with
                                     est=est_func,     
                                     param_name=param_name,  
                                     param_values=param_values, 
                                     snr=snr,         
                                     n_runs=n_runs    
                                     )

# testing the impact of parameter choice with signals sharing a non-linear relationship
kres_nlinear =  testing.test_est_params(comp1=evoked, 
                                        comp2=qua_signal, # Comparison is performed using the transformed signal.
                                        est=est_func,    
                                        param_name=param_name,  
                                        param_values=param_values, 
                                        snr=snr,         
                                        n_runs=n_runs    
                                        )

#%% Plotting the results of the simulation

# Creating the figure
fig, ax = plt.subplots(1,2, figsize=(10,5))
fig.suptitle(est_test)

# Plotting the results of the linear example
ax[0].set_title('linear')
ax[0].set_xlabel(param_name)
plotting.plot_param_results(kres_linear, kres_lower, ax=ax[0], color='r')

# Plotting the results of the non-linear example
ax[1].set_title('non-linear')
ax[1].set_xlabel(param_name)
plotting.plot_param_results(kres_nlinear, kres_lower, ax=ax[1], color='b')

#%%


#%% Imports

# Imports from our library
from micorr.simulation import simulations, testing, plotting
from micorr.estimators import mi_estimators, corr_est

# General imports
import numpy as np
from functools import partial

#%% Setting the initial parameters

# Dictionary with the estimators to be tested
est_list = {'PCC': corr_est.abs_correlation,
            'KDE': mi_estimators.kde_mi,
            'KSG': partial(mi_estimators.mi_ksg, k=5),
            'AB': mi_estimators.binned_mi}

# Simulating the base evoked response
sim_params = {'start':-0.05, 'end':0.3, 
              'peak_time':0.1,'peak_duration':0.12, 
              'sin_shift':0.01, 'fs':600}
evoked, times = simulations.simulate_evoked(**sim_params)

# Number of noise iterations to use
n_runs = 100

#%% SNR simulation

# SNRs to be tested
snrs_to_test = np.arange(-45, 45, 5) 
snr_results = testing.test_estimators(base_signal1=evoked, # Signal which SNR will be changed
                                      changes=snrs_to_test, # SNR values to be test  
                                      snr_db=None, # This needs to be set the None to change SNR across the iterations
                                      estimator_list=est_list, # Estimators to test
                                      n_runs=n_runs # Number of iterations to run 
                                      )

#%% Possibly saving the results of the SNR simulation 

save_path = None
if save_path:
    testing.save_sim_results(snr_results, save_path)
    
#%% Calculating the lower bound corresponding different SNRs

lower_bounds_snrs = {}
noise_std = np.std(evoked)
for snr in snrs_to_test: 
    print(f'Calculating: {snr}')
    add_lb = testing.test_estimators(base_signal1=evoked,
                                     compare_with_noise=noise_std, # Comparing the base signal with noise with the given std
                                     snr_db=snr, 
                                     estimator_list=est_list, 
                                     n_runs=n_runs,
                                     base_snr=True # SNR will be defined based on the base signal
                                     )
    lower_bounds_snrs[snr] = add_lb
    

#%% Saving the results of the lower bound simulation 

save_path = None
if save_path: 
    testing.save_sim_results(lower_bounds_snrs, save_path)
    
#%% Plotting the results of the simulation

# Parameters for the plotting 
y_labels = ['p', 'NMI', 'NMI', 'NMI']
x_label = 'SNR (dB)'
colors = [['#72E059', '#28B925', '#20830A'],
          ['#FFC04D', '#FFA500', '#FF8C00'],
          ['#6680E3','#103FF1','#45579B'],
          ['#E87C6E', '#F92306', '#B91F0A']]
estimators = list(est_list.keys())

# Plotting the results
plotting.plot_estimator_results(snr_results,
                                lower_bounds_snrs,
                                estimators,
                                y_labels, 
                                x_label,
                                colors, 
                                lower_type='adaptive' # We want to apply adaptive lower bound with this example
                                )

#%%
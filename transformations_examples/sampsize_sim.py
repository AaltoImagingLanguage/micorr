
#%% Imports

# Imports from the library
from micorr.simulation import simulations, testing, plotting, transformations
from micorr.estimators import mi_estimators, corr_est

# More general imports
import numpy as np
from functools import partial

#%% Initial parameters

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

# SNR of the added noise
snr = 20

# Whether to use sample size or sampling frequency as the changing parameter
paramater_change = 'freq'

#%% Downsampling

# Defining the function for downsampling
sampsize_func = partial(transformations.downsample_evoked, sim_params)

# Sampling frequency to downsample
fs_to_test = 20 

# Signals to compare after the downsampling
downsampled1 , downsampled2 = sampsize_func(evoked, new_fs=fs_to_test) 
downsampled_times = np.arange(sim_params['start'], sim_params['end'] , 1/fs_to_test)

# Plotting the downsampled signal 
plotting.plot_signals(downsampled1, downsampled2, downsampled_times, snr_aim=snr)
            
#%% Sample size simulation

# Sample sizes to test
samples_test = np.arange(6,350,20)
# Changing these sample sizes to frequencies
duration = sim_params['end'] - sim_params['start']
freqs = samples_test/duration


# Running the simulations
sampsize_results = testing.test_estimators(base_signal1=evoked, 
                                  changes=freqs, 
                                  trans_func=sampsize_func,
                                  snr_db=snr,
                                  estimator_list=est_list,
                                  n_runs=n_runs)

# Changing the keys to sampling frequency if this is set as the parameter key
if paramater_change == 'freq':
    sampsize_results = {estimator: {int(inner_key*duration): value for inner_key, value in inner_dict.items()}
                          for estimator, inner_dict in sampsize_results.items()}

#%% Possibly saving the results

save_path = None
if save_path:
    testing.save_sim_results(sampsize_results, save_path)

#%% Lower bound simulation

lower_bounds = {}
for freq in freqs: 
    samp_size = int(freq*duration)
    print(f'Calculating: {samp_size}')
    sim_params_new = {**sim_params, 'fs': freq}
    evoked, _ = simulations.simulate_evoked(**sim_params_new)
    noise_std = np.std(evoked)
    add_lb = testing.test_estimators(base_signal1=evoked,
                                     compare_with_noise=noise_std,
                                     snr_db=snr, 
                                     estimator_list=est_list, 
                                     n_runs=n_runs)
    lower_bounds[samp_size] = add_lb
    
#%% Saving the results

save_path = None
if save_path:
    testing.save_sim_results(lower_bounds, save_path)

#%% Plotting the results

# Parameters for the plotting 
y_labels = ['p', 'NMI', 'NMI', 'NMI']
x_label = 'fs'
colors = [['#72E059'],
          ['#FFC04D'],
          ['#6680E3'],
          ['#E87C6E']]
# Names corresponding to the estimators to plot
estimators = list(est_list.keys())

# Plotting the results
plotting.plot_estimator_results(sampsize_results,
                                lower_bounds,
                                estimators,
                                y_labels, 
                                x_label,
                                colors, 
                                lower_type='adaptive' # We want to apply adaptive lower bound with this example
                                )

#%%
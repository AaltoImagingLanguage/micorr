
#%%

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

#%% Transformation

# Controlling the time-shift (in samples)
shift = 30 
original, example_shifted = transformations.time_shift(evoked, shift)

# Plotting the two signals to be compared 
plotting.plot_signals(original, example_shifted, times, snr_aim=snr)

#%% Time-shift simulation

# Defining the shifts (in samples to test)
shifts = np.arange(0,150,10)
# Running the simualtion
shift_results = testing.test_estimators(base_signal1=evoked,
                                  changes=shifts,
                                  trans_func=transformations.time_shift,
                                  estimator_list=est_list,
                                  snr_db=snr,
                                  n_runs=n_runs)
# Changing the keys to seconds 
fs = sim_params['fs']
shift_results = {estimator: {(inner_key/fs): value for inner_key, value in inner_dict.items()}
                          for estimator, inner_dict in shift_results.items()}

#%% Possibly saving results

save_path = None
if save_path:
    testing.save_sim_results(shift_results, save_path)

#%% Simulating the lower bound

# With this example a constant lower bound will be applied to (because sample size and SNR of our base signal stay constant)
noise_std = np.std(evoked)
lower_bound = testing.test_estimators(base_signal1=evoked,
                                      compare_with_noise=noise_std,
                                      snr_db=snr, 
                                      estimator_list=est_list, 
                                      n_runs=n_runs)

#%% Possibly saving the lower bound

save_path = None
if save_path: 
    testing.save_sim_results(lower_bound, save_path)

#%% Plotting the results

# Parameters for the plotting 
y_labels = ['p', 'NMI', 'NMI', 'NMI']
x_label = '$\\Delta t$ (s)'
# Colors for the different estimators
colors = [['#72E059'],
          ['#FFC04D'],
          ['#6680E3'],
          ['#E87C6E']]

# Names corresponding to the estimators to plot
estimators = list(est_list.keys())

# Plotting the results
plotting.plot_estimator_results(shift_results,
                                lower_bound,
                                estimators,
                                y_labels, 
                                x_label,
                                colors, 
                                lower_type='constant')

#%%

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

#%% Transformation

# Transformation function to be used
change_dur = partial(transformations.change_dur_evoked,
                    sim_params=sim_params)

# An example case
new_dur = 0.05 
original, example_newdur = change_dur(evoked, new_dur)

# Plotting the two signals to be compared 
plotting.plot_signals(original, example_newdur, times, snr_aim=snr)

#%% Duration simulation

# Setting the durations to test
durs = np.arange(0.04,0.22,0.01)

# Running the simulation
duration_results = testing.test_estimators(base_signal1=evoked,
                                           changes=durs,
                                           trans_func=change_dur,
                                           estimator_list=est_list,
                                           snr_db=snr,
                                           n_runs=n_runs)

# Changing the keys into duration differences (in seconds) between the two signals
duration_results = {estimator: {np.round(inner_key-sim_params['peak_duration'],3): value for inner_key, value in inner_dict.items()}
                          for estimator, inner_dict in duration_results.items()}

#%% Possibly saving the results

save_path = None
if save_path: 
    testing.save_sim_results(duration_results, save_path)
    
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
x_label = '$\\Delta T$ (s)'
# Colors for the different estimators
colors = [['#72E059'],
          ['#FFC04D'],
          ['#6680E3'],
          ['#E87C6E']]
# Names corresponding to the estimators to plot
estimators = list(est_list.keys())

# Plotting the results
plotting.plot_estimator_results(duration_results,
                                lower_bound,
                                estimators,
                                y_labels, 
                                x_label,
                                colors, 
                                lower_type='constant')

#%%



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

# Defining the trasnsformation function
qua_func = partial(transformations.quadric_downsampling, sim_params)
# Applying the transformation
original, quadric = qua_func(evoked, new_fs=600)

# Plotting the two signals to be compared 
plotting.plot_signals(original, quadric, times, snr_aim=snr)

#%% Sample size simulation

# Defining the samples sizes to test (downsampling is done)
samples_test = np.arange(6,350,20)
duration = sim_params['end'] - sim_params['start']
freqs = samples_test/duration

#%% Running the simulation

qua_results = testing.test_estimators(base_signal1=evoked, 
                                      changes=freqs, 
                                      trans_func=qua_func,
                                      snr_db=snr,
                                      estimator_list=est_list,
                                      n_runs=n_runs)

#%% Possibly saving the results

save_path = None
if save_path:
    testing.save_sim_results(qua_results, save_path)
    
#%% Simulating the (adaptive lower bound)

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

#%% Possibly saving the lower bound

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
plotting.plot_estimator_results(qua_results,
                                lower_bounds,
                                estimators,
                                y_labels, 
                                x_label,
                                colors, 
                                lower_type='adaptive' # We want to apply adaptive lower bound with this example
                                )

#%%

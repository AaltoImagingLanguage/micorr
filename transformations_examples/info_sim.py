# %% Imports

# Imports from the library
from micorr.simulation import simulations, testing, plotting, transformations
from micorr.estimators import mi_estimators, corr_est

# More general imports
import numpy as np
from functools import partial

# %% Initial parameters

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

# Total number of samples in the simulated response
n_samples = len(evoked)

# %% Transformation

add_samp = 200  # How many data points are added
std_pad = 0.1  # How noisy padding will be used
padding_func = partial(transformations.add_padding, std_pad=std_pad)
padded_signal1, padded_signal2 = padding_func(evoked, add_samp)
padded_times = np.arange(
    sim_params['start'], sim_params['end'] + add_samp/sim_params['fs'], 1/sim_params['fs'])

# Plotting the two signals to be compared
plotting.plot_signals(padded_signal1, padded_signal2,
                      padded_times, snr_aim=snr)

# %% Running the simulation

# Defining the points additions to test
added_points = [1, 10, 30, 80, 200, 600, 1000, 2000, 4000, 8000]
# Defining the noise point standard deviations to test
std_multi = [0, 0.1, 0.5]

std_info_results = []
for multi in std_multi:
    std_pad = np.std(evoked)*multi
    padding_func = partial(transformations.add_padding, std_pad=std_pad)
    info_results = testing.test_estimators(base_signal1=evoked,
                                           changes=added_points,
                                           trans_func=padding_func,
                                           snr_db=snr,
                                           estimator_list=est_list,
                                           n_runs=n_runs
                                           )
    std_info_results.append(info_results)

# %% Possibly saving the results

save_path = None
if save_path:
    testing.save_sim_results(info_results, save_path)

# %% Simulating the adaptive lower bound (sample size changes)

info_lbounds = {}

# Lower bound
for add_points in added_points:
    print(f'Calculating: {add_points}')
    new_base, _ = transformations.add_padding(
        evoked, add_points, std_pad=std_pad)
    noise_std = np.std(new_base)
    add_lb = testing.test_estimators(base_signal1=new_base,
                                     compare_with_noise=noise_std,
                                     snr_db=snr,
                                     estimator_list=est_list,
                                     n_runs=n_runs
                                     )
    info_lbounds[add_points] = add_lb

#%% Saving the lower bound results

save_path = None
if save_path:
    testing.save_sim_results(info_lbounds, save_path)

#%% Plotting the simulation results

# Parameters for the plotting
y_labels = ['p', 'NMI', 'NMI', 'NMI']
x_label = '$N_{n}$'
# 2D color list: rows = estimators, columns = noisy point standard deviations
colors = [['#72E059', '#28B925', '#20830A'],
          ['#FFC04D', '#FFA500', '#FF8C00'],
          ['#6680E3', '#103FF1', '#45579B'],
          ['#E87C6E', '#F92306', '#B91F0A']]
# Names corresponding to the estimators to plot
estimators = list(est_list.keys())

# Plotting the results
plotting.plot_estimator_results(std_info_results,
                                info_lbounds,
                                estimators,
                                y_labels,
                                x_label,
                                colors,
                                lower_type='adaptive',  # Applying adaptive lower bound
                                axis_type='log' # Changing axis to logaritmic
                                )

# %%

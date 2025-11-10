
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

#%% Adding outliers

# Scaling the responses before adding outliers
scaled1, scaled2 = simulations.max_scaling(evoked, evoked)

# Cotrolling the number of outliers that will be added 
n_out = 20

# Adding outliers to one of the responses
outlier_response = transformations.add_outliers(scaled2, n_out=n_out)

# Plotting the two signals to be compared 
plotting.plot_signals(scaled1, outlier_response, times, snr_aim=snr)

#%%  Outlier simulation

# Defining the outlier numbers to test
n_samples = len(evoked)
n_outliers = np.arange(0,n_samples+15,15, dtype=int)
# Defining the z-scores to test
z_score_limits = [(7,9), (5,7), (3,5)]

# Running the simulation
results_outliers = []
for (l_z, u_z) in z_score_limits:
    results = testing.test_estimators(base_signal1=evoked, 
                                      changes=n_outliers, 
                                      outliers='one', #Outliers will be added one of the responses
                                      z_score=[l_z,u_z],
                                      estimator_list=est_list, 
                                      n_runs=n_runs, 
                                      snr_db=snr)
    results_outliers.append(results)

#%% Possibly saving the results 

save_path = None
if save_path:
    testing.save_sim_results(results_outliers, save_path)

#%% Lower bound

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
x_label = '$N_{o}$'
# 2D color list: rows = estimators, columns = z-scores
colors = [['#72E059', '#28B925', '#20830A'],
          ['#FFC04D', '#FFA500', '#FF8C00'],
          ['#6680E3','#103FF1','#45579B'],
          ['#E87C6E', '#F92306', '#B91F0A']]
# Names corresponding to the estimators to plot
estimators = list(est_list.keys())

# Plotting the results
plotting.plot_estimator_results(results_outliers,
                                lower_bound,
                                estimators,
                                y_labels, 
                                x_label,
                                colors, 
                                lower_type='constant')

#%%
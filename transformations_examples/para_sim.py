#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 12:18:37 2025

@author: hukaria2
"""

#%% Imports 

import sys
sys.path.append('/u/38/hukaria2/unix/mi_paper/micorr')
import estimators, simulations, testing, transformations

import numpy as np

#%% Simulating the base signals used in the comparision

sim_params = {'start':-0.05, 'end':0.3, 
              'peak_time':0.1,'peak_duration':0.12, 
              'sin_shift':0.01, 'fs':600}
evoked, times = simulations.simulate_evoked(**sim_params)
_, qua_signal = transformations.transf_power(evoked)
snr_aim = 20

#%% Impact of the choice of k

#ks = np.arange(1,20)
#ks = [int(k) for k in ks]
ks = [1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
ksg_est = estimators.mi_ksg

kres_linear = testing.test_est_params(evoked, evoked, ksg_est, 'k', ks, snr_aim, n_runs=1_000)
kres_lower = testing.test_est_params(evoked, np.std(evoked), ksg_est, 'k', ks, snr_aim, n_runs=1_000)
kres_nlinear =  testing.test_est_params(evoked, qua_signal, ksg_est, 'k', ks, snr_aim, n_runs=1_000)

#%% Saving the results

base_path = '/u/38/hukaria2/unix/mi_paper/simulation_results/freeparameter/'
testing.save_sim_results(kres_linear, base_path + 'kres_linear.pkl')
testing.save_sim_results(kres_lower, base_path + 'kres_lower.pkl')
testing.save_sim_results(kres_nlinear, base_path + 'kres_nlinear.pkl')

#%% Impact of the choice of N

bins = [1, 2, 3, 5, 10, 13, 15, 20, 25, 27, 30]  
binning_est = estimators.binned_mi

bres_linear = testing.test_est_params(evoked, evoked, binning_est, 'n_bins', bins, snr_aim)
bres_lower = testing.test_est_params(evoked, np.std(evoked), binning_est, 'n_bins', bins, snr_aim)
bres_nlinear = testing.test_est_params(evoked, qua_signal, binning_est, 'n_bins', bins, snr_aim)

#%% Saving the results 

testing.save_sim_results(bres_linear, base_path + 'bres_linear.pkl')
testing.save_sim_results(bres_lower, base_path + 'bres_lower.pkl')
testing.save_sim_results(bres_nlinear, base_path + 'bres_nlinear.pkl')

#%% Impact of the choice of bandwidth factor

#bfs = np.arange(0.1, 2, 0.2)  

bfs = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]
kde_est = estimators.kde_mi
bfres_linear = testing.test_est_params(evoked, evoked, kde_est, 'bw_method', bfs, snr_aim, n_runs=1_000)
bfres_lower = testing.test_est_params(evoked, np.std(evoked), kde_est, 'bw_method', bfs, snr_aim, n_runs=1_000)
bfres_nlinear = testing.test_est_params(evoked, qua_signal, kde_est, 'bw_method', bfs, snr_aim,n_runs=1_000)

#%%

base_path = '/u/38/hukaria2/unix/mi_paper/simulation_results/freeparameter/'
testing.save_sim_results(bfres_linear, base_path + 'bfres_linear.pkl')
testing.save_sim_results(bfres_lower, base_path + 'bfres_lower.pkl')
testing.save_sim_results(bfres_nlinear, base_path + 'bfres_nlinear.pkl')

#%% 
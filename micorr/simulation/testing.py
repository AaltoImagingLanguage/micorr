#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 12:21:16 2025

@author: anni
"""


from . import transformations, simulations
from ..snr import snr_func
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle

def test_estimators(base_signal1,
                    base_signal2=None,
                    scaling_func=simulations.max_scaling,
                    changes=None, 
                    trans_func=None,  
                    outliers= None,
                    z_score = [3,5],
                    estimator_list=None,
                    compare_with_noise=False, 
                    n_runs=1000, 
                    snr_db=10,
                    add_noise=[1,1],
                    base_snr=False):
    '''
    
    Test similarity estimatore by applying transformations, adding noise, or adding outliers to a base signal. 
    
    Only one of the following can be used per function call:
        - trans_function: to ba a transformation to the signal. 
        - outliers: to add outlier to the signal.
        - compare_with_noise: to compare the signal with pure noise.
        - `snr_db = None`: to vary the signal-to-noise ratio across runs.

    These options are mutually exclusive. If more than one is specified, a ValueError is raised.
    
    Parameters
    ----------
    base_signal : np.array
        A noise free base signal that will be transformed.
    scaling_func: callable or None
        A function used to scale the signals.
        if None no scaling is applied.
        By default scaling based on the shared maximum value of the two signals is applied. 
    changes : list
        A list of transformation values to test
        (e.g., noise levels, transformation magnitudes, number of outliers).
    trans_function : callable or None, optional
         A function applied the transform the signal.
         The provided function should return both two signals to be compared.
         If None no transformation applied. The default is None.
    outliers : None or lstr
        Whether outliers are added.
        If None no outliers are added.
        If 'one' outliers are added into base_signal1. 
        If 'both'outliers are added into both of the signals. 
        The default is None.
    compare_with_noise: bool or int
        If a float is provided, the signal is compare with a  pure noise signal with that standard deviation.
        If False (default) noise comparision is not done.
    n_runs : int, optional
        Number of iteration to run the simulation. The default is 1000.
    snr_db : int, optional
        The signal to noise ratio of the simulated signal after the random noise is added. 
        If None the noise is changed based on the changes parameter.
        The default is 10.
    add_noise: list, optional
        Whether the noise is added into both of the signals or only first of the signals.
        One in the list means that the noise will be added into that signal. 
    base_snr: bool, optional
        If True the SNR will de defined based on the given base signal.
        If False (default) the transformed signal will be used to define the SNR
        
    Returns
    -------
    results : dict
        A dictionary containing the results of the given similarity estimators 
        across iterations where random noise is added 
        while changing the given transformation parameter.
        
    '''
    
    # Ensuring that the user tries to do only one of the options
    mode_flags = [
        trans_func is not None,
        outliers is not None,
        compare_with_noise not in [False, None],
        snr_db is None]

    if sum(mode_flags) > 1:
        raise ValueError("Only one of the following can be used at a time: trans_function, outliers, "
                         "compare_with_noise, or setting snr_db=None.")

    # Initializing dictionaries for the estimator results
    if changes is not None:
        results = {estimator: {} for estimator in estimator_list}
    else:
        changes = [None]
        results = {estimator: [] for estimator in estimator_list}  
    
    if base_signal2 is None:
        base_signal2 = base_signal1
    
    if base_snr:
        before_trans = base_signal1
        before_trans_scaled, _ = scaling_func(before_trans, before_trans)
    
    # Tracking the progress of the calculations
    n_total = len(changes) * n_runs * len(estimator_list)
    with tqdm(total=n_total, desc='Total Progress') as pbar:
        
        # Looping through the different values applied in the transformations
        for change in changes:
            
            # transformation applied across the changes
            if trans_func:
                transformed1, transformed2 = trans_func(base_signal1, change)
            else:
                transformed1, transformed2 = base_signal1, base_signal2
            
            # Scaling
            if scaling_func:
                scaled1, scaled2 = scaling_func(transformed1, transformed2)
        
            # Adding random noise in n_runs iterations
            for i in range(n_runs): 
                
                # When snr_db is spet to None we change the SNR level
                snr_aim = snr_db if snr_db is not None else change
                
                # Adding noise into base signal
                if add_noise[0] == 1:
                    if base_snr == False:
                        noise1, _ = snr_func.snr_to_noise(snr_aim, scaled1)
                    else:
                        noise_std = snr_func.snr_to_std(snr_db, before_trans_scaled)
                        noise1 = np.random.normal(0, noise_std, size=len(scaled1))
                    noisy_signal1 = scaled1 + noise1
                else:
                    noisy_signal1 = scaled1
                
                # Do we compare with the noise signal
                if compare_with_noise:
                    noise_std = np.max(scaled1)
                    noisy_signal2 = np.random.normal(0, noise_std, size=len(noisy_signal1))
                    
                # We compare with the transformed signal
                else:
                    # If it is defined that the noise is added into both of the signals
                    if add_noise[1] == 1:
                        if base_snr == False:
                            noise2, _ = snr_func.snr_to_noise(snr_aim, scaled2)
                        else:
                            noise_std = snr_func.snr_to_std(snr_db, before_trans_scaled)
                            noise2 = np.random.normal(0, noise_std, size=len(scaled1))
                        noisy_signal2 = scaled2 + noise2
                    else:
                        noisy_signal2 = scaled2
                
                # Possible outliers are added 
                if outliers in ('one', 'both'):
                    noisy_signal2 = transformations.add_outliers(noisy_signal2, change, zscore_range=z_score)
                    if outliers == 'both':
                        noisy_signal1 =  transformations.add_outliers(noisy_signal1, change, zscore_range=z_score)
                
                # Calculating the results with the different estimators
                for estimator in estimator_list:
                    estimator_function = estimator_list[estimator]
                    estimator_value = estimator_function(noisy_signal1, noisy_signal2)
                    # Adding change as a key when changes are used
                    if change is not None:
                        results[estimator].setdefault(change, []).append(float(estimator_value))
                    else:
                        results[estimator].append(estimator_value)
                    pbar.update(1)
                    
    return results

def test_est_params(comp1, comp2, 
		    est, param_name, param_values, 
		    snr, n_runs=1_000, scaling_func=simulations.max_scaling):
    '''
    Testing how the choice of freeparameter impacts on the results of the different MI estimators.
    
    Parameters
    ----------
    comp1: np.array
    	First signal to compare with the given estimator.
    comp2: np.array or float
    	Second signal to compare with the given estimator. 
    	If a float is provided, a random noise signal is generated with the specified
	standard deviation (the float value) and the same length as `comp1`, 
	and this noise signal is used as the second signal.
    est: callable
    	Function of the estimator the be tested.
    param_name:
    	The name of the freeparameter ti be changed in the given estimator function.
    param_values: list
    	List of the free parameter values to test.
    snr: int
    	Noise level (dB) to add in the given signals. 
    n_runs: int, optional
    	Number of iteration to run the simulation. The default is 1000.
    scaling_func: callable, optional
    	A function used to scale the signals.
        if None no scaling is applied.
        By default scaling based on the shared maximum value of the two signals is applied. 
    
    Returns
    -------
    param_results : dict
    	A dictionary containing the results of the given similarity estimator 
        across iterations where random noise is added 
        while changing the given free parameter of the estimator.
    
    '''
    if scaling_func:
    	comp1, comp2 = scaling_func(comp1, comp2)
    
    param_results = {}
    n_total = n_runs * len(param_values)
    with tqdm(total=n_total, desc='Total Progress') as pbar:
        for param in param_values:
            est_results = []
            for _ in range(n_runs):
                n1, _ = snr_func.snr_to_noise(snr, comp1)
                comp1_noise = comp1 + n1
                if isinstance(comp2, (float)):
                    comp2_noise = np.random.normal(0, comp2, size=len(comp1))
                else:
                    n2, _ = snr_func.snr_to_noise(snr, comp2)
                    comp2_noise = comp2 + n2
                est_result = est(comp1_noise, comp2_noise, **{param_name: param})
                est_results.append(est_result)
                pbar.update(1) 
            param_results[param] = est_results
            
    return param_results


def save_sim_results(results, save_path):
    with open(save_path, 'wb') as f: 
        pickle.dump(results, f)
        
def get_sim_results(data_path):
    with open(data_path, 'rb') as f:
        results = pickle.load(f)
    return results


    


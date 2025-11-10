
'''
This file contains all of the implemented tranfromations. 
'''

import numpy as np
from . import simulations
import random


def transf_amplitude(time_series, scale_factor):
    # TO FIX: Add docs string in here
    scaled = np.dot(scale_factor, time_series)
    return time_series, scaled


def time_shift(time_serie, samples_shift):
    
    '''
    Given time-series is shifted (forward) by the given number of samples.
    Parameters
    ----------
    time_serie : np.array
        Time series to be shifted
    samples_shift : int
        The number of samples by the time series is shifted.

    Returns
    -------
    The shifted time series

    '''
    
    shifted_signal = np.roll(time_serie, samples_shift)
    return time_serie, shifted_signal
    
def transf_power(time_serie):
    '''
    Gives the square of the given time-series.
    Parameters
    ----------
    time_series : np.array
        Time series to be transformed.

    Returns
    -------
    np.array
        Square of the given time-series.

    '''
    power_serie = np.power(time_serie, 2)
    return time_serie, power_serie


# TO FIX: I am not 100% If we need to create an function out of this
def downsample_evoked(sim_params, time_series, new_fs):
    '''
    Downsampling evoked response 
    
    Parameters
    ----------
    sim_params : dict
        A dictionary containing the parameter used in the simulation of the base avoked response.
    new_fs : int
        New sampling frequency of the evoked response

    Returns
    -------
    evoked : np.array
        Evoked response simulated with the proovided parameters.

    '''
   
    
    downsampled, _ = simulations.simulate_evoked(start=sim_params['start'], 
                                                 end=sim_params['end'],
                                                 fs=new_fs, 
                                                 peak_time=sim_params['peak_time'],
                                                 peak_duration=sim_params['peak_duration'],
                                                 sin_shift=sim_params['sin_shift'])
    return downsampled, downsampled


def add_outliers(signal_to_add, n_out, zscore_range=[3,5]):
    '''
    Adding outliers in the given z_score range in the given signal.
    ----------
    signal_to_add : np.array
        A signal where the outlier are added.
    n_out : int
        A number of added outliers
    lower_zscore : int, optional
        A lower limit of the z-score. The default is 3.
    upper_zscore : int, optional
        A upper limit of the z-score. The default is 5.

    Returns
    -------
    signal_to_add : np.array
        A signal where the outliers are added.

    '''
    
    # Checking that a suitable number of outliersis given 
    if n_out > len(signal_to_add):
        raise ValueError('The given number of outliers is larger than the length of the signal')
    
    changed_index = []
    signal_std = np.std(signal_to_add)
    signal_mean = np.mean(signal_to_add)
    for _ in range(n_out):
        outlier_index = np.random.randint(0,len(signal_to_add))
        while outlier_index in changed_index:
            outlier_index = np.random.randint(0,len(signal_to_add))
        changed_index.append(outlier_index)
        original_value = np.abs(signal_to_add[outlier_index])
        lower_lim =  (zscore_range[0] * signal_std) - (original_value - signal_mean)
        upper_lim = (zscore_range[1] * signal_std) - (original_value - signal_mean)
        outlier_value =  np.random.uniform(lower_lim, upper_lim)
        sign = random.choice(['+','-'])
        if sign == '+':
            signal_to_add[outlier_index] += outlier_value
        else:
            signal_to_add[outlier_index] -= outlier_value
            
    return signal_to_add

# TO FIX: Add description
def add_padding(signal, len_pad, std_pad):
    if std_pad == 0:
        padding1 = np.zeros(len_pad)
        padding2 = np.zeros(len_pad)
    else:
        padding1 = np.random.normal(0, std_pad, size=len_pad)
        padding2 = np.random.normal(0, std_pad, size=len_pad)
    padded_signal1 = np.concatenate((signal.copy(), padding1))
    padded_signal2 = np.concatenate((signal.copy(), padding2))
    return padded_signal1, padded_signal2

# TO FIX: Add description
def quadric_downsampling(sim_params, time_series, new_fs):
    downsampled, _ = downsample_evoked(sim_params, time_series, new_fs)
    _ , quadric = transf_power(downsampled)
    return downsampled, quadric

# TO FIX: Add description
def change_dur_evoked(base, new_dur, sim_params):
    
    transformed, _ = simulations.simulate_evoked(start=sim_params['start'], 
                                                 end=sim_params['end'],
                                                 fs=sim_params['fs'], 
                                                 peak_time=sim_params['peak_time'],
                                                 peak_duration=new_dur,
                                                 sin_shift=sim_params['sin_shift'])
    return base, transformed


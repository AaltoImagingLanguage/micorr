
'''This file contains functions that can be applied to discrete signals into bins with different strategies.'''

import numpy as np 

def adaptive_binning(source, n_bins):
    '''
    Binning the signal into bins with equal number of data points (adaptive binning).
    
    Parameters
    ----------
    source : np.ndarray
        The input signal to be discretized into bins.
    n_bins : int
        The number of bins to apply in the binning.
        
    Returns
    -------
    binned_array : np.ndarray
        Binned signal
    '''
    spacing = np.linspace(0, source.size, n_bins + 1)
    #This give the inices sorted by magnitude of the values
    sorted_indices = np.argsort(source)
    binned_array = np.empty(len(source))
    for (i, lim) in enumerate(spacing):
        index = int(lim)
        if index == 0:
            previous = index
        else: 
            indices = sorted_indices[previous:int(lim)]
            binned_array[[indices]] = np.full((len(indices)), i)
            previous = int(lim)
    return binned_array

def equal_binning(source, n_bins):
    
    '''
    
    Binning the signal into equally spaced bins. 

    Parameters
    ----------
    source : np.ndarray
        The input signal to be discretized into bins.
    n_bins : int
        The number of bins to apply in the binning 
        
    Returns
    -------
    binned_array : np.ndarray
        Binned signal

    '''
    binned_array = np.zeros(source.size)
    #limits with equal width
    limits = np.linspace(min(source), max(source), n_bins, endpoint = False)
    #First
    previous = min(source)
    #For naming the bins
    bin_name = 0
    #going through all of the limits
    for lim in limits:
        index = 0
        for value in source:
            if previous <= value < lim:
                binned_array[index] = bin_name
                index = index +  1 
        bin_name = bin_name + 1
        previous = lim
    return binned_array




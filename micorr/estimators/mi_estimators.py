
'''This file contains functions that can be applied to calculate normalized mutual information '''

import numpy as np
import ennemi
import scipy
import math
from scipy.stats import gaussian_kde
from sklearn.metrics import mutual_info_score

from . import binning

def gaussian_mi(s1, s2, normalize=True):
    """
    Estimating the pairwise mutual information between two signals when both of the signals have gaussian distribution.
    
    Parameters
    ----------
    signal1: np.ndarray 
        The first signal to compare.
    signal2: np.ndarray 
        The second signal to compare.
    normalize: bool, optional
        If True (default), the estimated mutual information value is normalized.
        
    Returns
    -------
    mi: float
        The estimated normalized or non-normalized mutual information (in nats).
    """
    pearson, _ = scipy.stats.pearsonr(s1,s2)
    power_value = math.pow(pearson,2)
    if power_value == 1:
        power_value  = 0.99
    mi = -0.5*(np.log((1-power_value),2))
    if normalize:
        mi = np.sqrt(1-np.exp(-2*abs(mi)))
    return mi


def mi_ksg(signal1, signal2, k=3, normalize=True):
    """
    Estimating the pairwise mutual information between two signals using KSG estimator.
    Ennemi library developed by Laarne et al. 2021 is applied in the estimation. 
    
    Parameters
    ----------
    signal1: np.ndarray
        The first signal to compare.
    signal2: np.ndarray
        The second signal to compare.
    k: int, optional
        The number of neighbors KSG estimator uses. The default is 3.
    preprocess: bool, optional
        Defines if the values are scaled to unit variance before applying 
        KSG estimator. If False (default) the values are not scaled before
        preprocessing. 
    normalize: bool, optional
        If True (default), the estimated mutual information value is normalized.
        
    Returns
    -------
    mi: float 
        The estimated normalized or non-normalized mutual information (in nats).
    """
    
    signal1 = (signal1 - np.mean(signal1)) / np.std(signal1)
    signal2 = (signal2 - np.mean(signal2)) / np.std(signal2)
    mi = ennemi.estimate_mi(signal1, signal2, normalize=normalize, k=k, preprocess=False)[0][0]
    return np.abs(mi)


def kde_mi(signal1, signal2, normalize=True, bw_method='scott'):
    
    '''
    Estimating the pairwise mutual information between two signals using KDE estimator with Gaussian kernel 
    where the bandwidth of the kernel is defined with Scott's rule. 
    
    Parameters
    ----------
    signal1: np.ndarray
        The first signal to compare.
    signal2: np.ndarray
        The second signal to compare.
    normalize: bool
        If True (default), the estimated mutual information value is normalized.
    Returns
    -------
    mi: float 
        The estimated normalized or non-normalized mutual information (in nats).
    '''

    if isinstance(bw_method, float):
        if int(bw_method) == 1:
            bw_method1d = 'scott'
            bw_method2d = 'scott'
        else:
            n = len(signal1)
            bw_method1d = bw_method * (n ** (-1 / (1 + 4)))
            bw_method2d = bw_method * (n ** (-1 / (2 + 4)))
    else:
        bw_method1d = bw_method
        bw_method2d = bw_method
  
    dataset = np.stack((signal1, signal2)) 
    kde_X = gaussian_kde(dataset[0,:], bw_method=bw_method1d)
    kde_Y = gaussian_kde(dataset[1,:], bw_method=bw_method1d)
    kde_XY = gaussian_kde(dataset, bw_method=bw_method2d)
    
    
    x_pdf = kde_X(dataset[0,:])
    y_pdf = kde_Y(dataset[1,:])
    xy_pdf = kde_XY(dataset)
    
    epsilon = 1e-10
    x_pdf = np.clip(x_pdf, epsilon, None)
    y_pdf = np.clip(y_pdf, epsilon, None)
    xy_pdf = np.clip(xy_pdf, epsilon, None)
    
    # MI is calculated through the entropies
    h_XY = -np.mean(np.log(xy_pdf))
    h_X = -np.mean(np.log(x_pdf))
    h_Y = -np.mean(np.log(y_pdf))
    mi = (h_X + h_Y - h_XY)
    
    # Knowing entropies allows testing large range of normalization strategies
    if normalize:
        if mi > 0:
            mi = np.sqrt(1 - np.exp(-2 * mi))
        else:
            mi = 0
    return mi

def binned_mi(s1, s2, n_bins=None, bias_correction=True, bin_type='adaptive', normalize=True):
    
    '''
     Estimating mutual information between two signals by binning the signals.
     
     Parameters
     ----------
     signal1: np.ndarray
         The first signal to compare.
     signal2: np.ndarray
         The second signal to compare.
    n_bins: int, optional
        Number of bins used in the binning. 
        If None (default) Doanes rule is applied to pick the number of bins. 
    bias_correction: str, optional
        Defines whether bias correction is applied. 
        If True (default) bias correction is applied.
    bin_type: str, optional
        Defines how the binning is done. 
        If 'equal' equal sized bins are used.
        If 'adaptive' (default) adaptive binning is used. 

     Returns
     -------
     mi: float
         The estimated normalized or non-normalized mutual information (in nats).
     
     '''
    if bin_type not in ['adaptive', 'equal']:
        raise ValueError(f"Invalid bin_type: {bin_type}. Expected 'adaptive' or 'equal'.")
    if n_bins == None:
        n_bins1 = len(np.histogram_bin_edges(s1,'doane'))
        n_bins2 = len(np.histogram_bin_edges(s2,'doane'))
        n_bins = np.min([n_bins1, n_bins2])
        #print(f'{n_bins} used in the binning')
    if bin_type == 'adaptive':    
        b1 = binning.adaptive_binning(s1, n_bins)
        b2 = binning.adaptive_binning(s2, n_bins)
    elif bin_type == 'equal':
        b1 = binning.equal_binning(s1, n_bins)
        b2 = binning.equal_binning(s2, n_bins)
    mi = mutual_info_score(b1, b2)
    if mi != 0:
        if bias_correction == True:
            assert s1.shape == s2.shape
            n = s1.shape[0]
            ptc = (n_bins-1)**2 / (2*n)
            mi -= ptc
        if mi > 0: 
            if normalize:
                nmi = np.sqrt(1-np.exp(-2*mi))
            else:
                nmi = mi 
        else:
            nmi = 0
    else:
        nmi = 0
    return nmi


 

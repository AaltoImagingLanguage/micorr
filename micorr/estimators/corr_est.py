import numpy as np
from scipy.stats import pearsonr

def abs_correlation(signal1, signal2):
    '''
     Calculating absolute value of Pearson correlation between two signals.
     
     Parameters
     ----------
     signal1: np.ndarray
         The first signal to compare.
     signal2: np.ndarray
         The second signal to compare.

     Returns
     -------
     abs_correlation: float
         The absolute value of Pearson correlation.
     
     '''
    abs_corr = np.abs(pearsonr(signal1, signal2)[0])
    
    return abs_corr
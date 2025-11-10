
from tqdm import tqdm
import numpy as np
import mne

from micorr.snr import  snr_func

# Testing given estimators by comparing single channels response with all other channels
def cal_rel_values(est_list, sig_comp, data, snr=30):

    results = {est: [] for est in est_list}
    
    n_total = data.shape[0] * len(est_list) 
    with tqdm(total=n_total, desc='Total Progress') as pbar:
        if snr != None:
            comp_std = snr_func.snr_to_std(snr, sig_comp)
            sig_comp_noise = sig_comp + np.random.normal(0, comp_std, size=len(sig_comp))
        else:
            sig_comp_noise = sig_comp
        for ch_data in data:
            resp = ch_data.flatten()
            if snr != None:
                noise_std = snr_func.snr_to_std(snr, resp)
                resp_noise = resp + np.random.normal(0, noise_std, size=len(resp))
            else:
                resp_noise = resp
            for est in est_list:
                estimator = est_list[est]
                value = estimator(resp_noise, sig_comp_noise)
                results[est].append(value)
                pbar.update(1)
    
    return results


# Estimating the lower bound for the given estimators by comparing given singal with a random noise
def lowerbound_random(est_list, sig_comp, snr=30, n_runs=10_000):
    
    lower_random = {est: [] for est in est_list}
    
    n_total = n_runs * len(est_list) 
    with tqdm(total=n_total, desc='Total Progress') as pbar:
        for i in range(n_runs):
            if snr != 0:
                noise_std_sig = snr_func.snr_to_std(snr, sig_comp)
                noisy_signal = sig_comp + np.random.normal(0, noise_std_sig, len(sig_comp))
            else:
                noisy_signal = sig_comp
            noise = np.random.normal(0, np.std(noisy_signal), size=len(sig_comp))
            for est in est_list:
                estimator = est_list[est] 
                value = estimator(noisy_signal, noise)
                lower_random[est].append(value)
                pbar.update(1) 
    
    return lower_random

# Ordering the MEG channels based on the magnitude of the estimator values
def order_magnitude(est_res, info, ch_type):
    
    # Choosing the correct channels
    if ch_type == 'all':
        picks_meg = mne.pick_types(info, meg=True)
    elif ch_type == 'grad':
        picks_meg = mne.pick_types(info, meg='grad')
        exclude_meg = mne.pick_types(info, meg='mag')
    elif ch_type == 'mag':
        picks_meg = mne.pick_types(info, meg='mag')
        exclude_meg = mne.pick_types(info, meg='grad')
    elif ch_type == 'planar1':
        picks_meg = mne.pick_types(info, meg='planar1')
        exclude1 = mne.pick_types(info, meg='planar2')
        exclude2 = mne.pick_types(info, meg='mag')
        exclude_meg = np.concatenate([exclude1, exclude2])
    elif ch_type == 'planar2':
        picks_meg = mne.pick_types(info, meg='planar2')
        exclude1 = mne.pick_types(info, meg='planar1')
        exclude2 = mne.pick_types(info, meg='mag')
        exclude_meg = np.concatenate([exclude1, exclude2])
    
    # Removing the wrong channeös
    ch_names = [info['ch_names'][i] for i in picks_meg]  
    if ch_type in ['grad', 'mag', 'planar1']:
        est_res = np.delete(est_res, exclude_meg)
    
    # Sorint the channels
    sorted_idx = np.argsort(est_res)
    sorted_names = [ch_names[i] for i in sorted_idx]
    sorted_values = [est_res[i] for i in sorted_idx]
    
    return sorted_idx, sorted_names, sorted_values


# Calculating the p-value for each estimator based on the lower bound
def calc_pvals(results, lower):
    
    p_vals = {est:[] for est in results}

    for est in results:
        for val in results[est]:
            p_val = cal_pval(val, lower[est])
            p_vals[est].append(p_val)
    return  p_vals

# Calculating a single p-valyes based on the lower bound
def cal_pval(val, lower):
    lower = np.array(lower).flatten() 
    p_val = (np.sum(lower >= val) + 1)/ (len(lower))  
    return p_val

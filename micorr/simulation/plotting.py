'''This file contains functions that can be applied to plot the simulated signals and the results of the simulations'''

import matplotlib.pyplot as plt 
import numpy as np 
from ..snr import snr_func
from . import simulations
from matplotlib import gridspec
from matplotlib.ticker import LogLocator, LogFormatterSciNotation

def plot_estimator_results(results, 
                           lower,
                           estimators,
                           y_labels, 
                           x_label,
                           colors, 
                           labels=False, 
                           lower_type='constant',
                           axis_type='linear'):
    
     n_est = len(estimators)
     fig = plt.figure(figsize=(10*n_est, 4))
     outer = gridspec.GridSpec(1, n_est, wspace=0.4)
    
     for i, est in enumerate(estimators):
         
         if lower_type == 'constant':
             inner = gridspec.GridSpecFromSubplotSpec(1, 2, 
                                                      subplot_spec=outer[i], 
                                                      width_ratios=[10, 2], 
                                                      wspace=0.05)
             ax_main = fig.add_subplot(inner[0, 0])
             ax_hist = fig.add_subplot(inner[0, 1], sharey=ax_main)
             
         if lower_type == 'adaptive':
             inner = gridspec.GridSpecFromSubplotSpec(1, 1, 
                                                      subplot_spec=outer[i])
             ax_main = fig.add_subplot(inner[0])
             
         # Setting the frame for the main figure
         ax_main.set_ylabel(y_labels[i], fontsize=10)
         ax_main.set_ylim([-0.1,1.1])
         ax_main.set_xlabel(x_label)
         
         # Ensuring that the results are inside a list
         if not isinstance(results, list):
             results = [results]
         
         # Plotting the estimator results
         for j, single_results in enumerate(results):
             if j == 0:
                 ax_main.set_title(est)
             results_est = single_results[est]
             avs = [np.median(results_est[key]) for key in results_est.keys()]
             plot_errorbars(results_est, avs, colors[i][j], ax_main)
             
             # Labels are added if they are given
             if labels:
                 ax_main.plot(list(results_est.keys()), avs,  c=colors[i][j],
                         linestyle='dashed', linewidth=1,
                         markersize=8, marker='.',
                         alpha=0.5, label=labels[j])
                 ax_main.legend(fontsize=10)
             else:
                 ax_main.plot(list(results_est.keys()), avs,  c=colors[i][j],
                         linestyle='dashed', linewidth=1,
                         markersize=8, marker='.',
                         alpha=0.5)
         
         # If the lower bound stays constant
         if lower_type == 'constant':
             
             # P-value addition
             pval_lim = np.percentile(lower[est], 99)
             ax_main.axhline(y=pval_lim, color='k', linewidth=1, alpha=0.5, linestyle='dashed')
             ax_main.axhline(y=np.median(lower[est]), color='grey', linewidth=1, alpha=0.5, linestyle='dashed')
             # Here, I need to change the limits
             x_vals = list(results[0][est].keys())
             ax_main.fill_between([x_vals[0], x_vals[-1]], np.max(lower[est]), np.min(lower[est]), color='grey', alpha=0.2)
             
             # Adding a histogramin the side
             ax_hist.hist(lower[est], bins=30, orientation='horizontal', color=colors[i][0], edgecolor='none', alpha=0.7)
             ax_hist.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
             for spine in ['top', 'right', 'bottom', 'left']:
                 ax_hist.spines[spine].set_visible(False)
         
         # If we have adaptive lower bound
         if lower_type == 'adaptive':
             
             # Getting the limits
             upper_lims = [np.max(lower[key][est]) for key in lower]
             lower_lims = [np.min(lower[key][est]) for key in lower]
             median_vals = [np.median(lower[key][est]) for key in lower]
             p_vals = [np.percentile(lower[key][est], 99) for key in lower]
                 
             # Fill the area between the two lines
             x_vals = list(results[0][est].keys())
             ax_main.plot(x_vals, median_vals,  color='grey', linewidth=1, alpha=0.5, linestyle='dashed')
             ax_main.plot(x_vals, p_vals, color='k', linewidth=1, alpha=0.5, linestyle='dashed')
             ax_main.fill_between(x_vals, lower_lims, upper_lims, color='grey', alpha=0.2)
        
        
         if axis_type == 'linear':
            ax_main.grid(which='both', linestyle='--', alpha=0.5)
            ax_main.tick_params(axis='both', which='major', labelsize=10)
       
         if axis_type == 'log':
            ax_main.set_xscale('log')
            ax_main.xaxis.set_major_locator(LogLocator(base=10))
            ax_main.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)*0.1, numticks=100))
            ax_main.xaxis.set_major_formatter(LogFormatterSciNotation())
            ax_main.tick_params(axis='x', which='major', length=6, labelsize=10)
            ax_main.tick_params(axis='x', which='minor', length=3, labelsize=10)
            ax_main.tick_params(axis='y', which='major', labelsize=10)
            ax_main.grid(which='both', linestyle='--', alpha=0.5)
         
         
# Adding error bars in the results figure
def plot_errorbars(results, avs, c1, ax):
    
    max_values = [np.max(results[key]) for key in results.keys()]
    min_values = [np.min(results[key]) for key in results.keys()]
    errors_lower = [avs[i] - min_values[i] for i in range(len(avs))]
    errors_upper = [max_values[i] - avs[i] for i in range(len(avs))]
    ax.errorbar(list(results.keys()), avs, yerr=[errors_lower, errors_upper], fmt='--', 
                color=c1, capsize=4, alpha=0.4, capthick=1.2)
   
         
def plot_signals(signal1, signal2, times,
                 snr_aim=10, 
                 save_path=None, 
                 scal_func=simulations.max_scaling,
                 c1='black', c2='darkgrey',
                 title=None, ax=None):
    
    '''
    Plots an example figure of the two signals with the given signal-to-noise ratio (SNR).
        
    Parameters
    ----------
    signal1 : np.ndarray
        The first signal to plot
    signal2 : np.ndarray
        The second signal to plot
    times : np.ndarray
        Time vector
    snr_aim : int, optional
        Signal-to-noise ratio (SNR) of the signals. The default is 10.
    save_path : str, optional
        If a path is provided as a string, the figure is saved at that path. 
        If None (default), the figure is not saved.
    scal_func: Callable or None, optional
        The funstion applied to scale the simulate signals.
        If None no scaling is applied
    c1: str, optional
        Color used with the first signal.
    c2: str, optional
        Color used with the second signal.
    title: str, optional
        The title of the figure.
        If None (default), no title is added.
    ax: plt.Axes or None, optional
        Where the figure is plotted.
        If None new figure is formed.

    Returns
    -------
    None.

    '''
    
    # Possibly scaling the signals
    if scal_func:
        signal1_scaled, signal2_scaled = scal_func(signal1, signal2)
    
    # Defining the noise deviation based on the aim SNR
    noise1, noise_std1 = snr_func.snr_to_noise(snr_aim, signal1_scaled)
    noise2, noise_std2 = snr_func.snr_to_noise(snr_aim, signal2_scaled)
    
    # Forming the figure if Axes to plot is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize = (10,5))
    
    noisy_signal1 = signal1_scaled + noise1
    noisy_signal2 = signal2_scaled + noise2
    ax.plot(times, noisy_signal1, c=c1)
    ax.plot(times, noisy_signal2, c=c2)

    ax.tick_params(axis='both', labelsize=10) 
    ax.set_ylabel('amplitude (a.u.)', fontsize=12)
    ax.set_xlabel('time (s)', fontsize=12)
    [ax.spines[side].set_visible(False) for side in ['right', 'top']]
    ax.grid(axis = 'x', linestyle = '--')
    ax.axhline(y=0, linewidth=1, linestyle='dashed', color='grey', alpha=0.5)
    
    # Adding a title if one is given
    if title is not None:
        ax.set_title(title, fontsize=12)
        
    # Possibly saving the figure
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
    
# TO DO: write the desciption in here, maybe add comments
def plot_param_results(results, lower, ylab='NMI', color='k', ax=None, chosen=None):
    
    if ax is None:
    	fig, ax = plt.subplots()
    avs = [np.median(results[key]) for key in results.keys()]
    plot_errorbars(results, avs, color, ax)
    ax.plot(list(results.keys()), avs,  c=color,
            linestyle='dashed', linewidth=1,
            markersize=8, marker='.',
            alpha=0.5)
            
    upper_lims, lower_lims, median_vals, p_vals = [], [], [], []
    for v in lower.values():
        lower_lims.append(np.min(v))
        upper_lims.append(np.max(v))
        median_vals.append(np.median(v))
        p_vals.append(np.percentile(v, 99))
	
    ax.set_ylim([-0.1,1.1])
    ax.set_ylabel(ylab)
    ax.tick_params(axis='both', labelsize=9)
    
    x_vals = list(results.keys())
    ax.plot(x_vals, median_vals,  color='grey', linewidth=1, alpha=0.5, linestyle='dashed')
    ax.plot(x_vals, p_vals, color='k', linewidth=1, alpha=0.5, linestyle='dashed')
    ax.grid(which='both', linestyle='--', alpha=0.5)
    ax.fill_between(x_vals, lower_lims, upper_lims, color='grey', alpha=0.2)
    
    if chosen:
        ax.axvline(x=chosen, color='darkred', linestyle='--', linewidth=1) 
         



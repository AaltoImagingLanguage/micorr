
'''This file contains the plotting functios that can be used when testing the estimators with real MEG data'''

# General imports
import numpy as np 
import matplotlib
from matplotlib import gridspec
import matplotlib.pyplot as plt
import mne

# Imports from our library
from micorr.realdata import ch_locs, test_est
from micorr.simulation import simulations

# TO DO: Here doc strings are still missing

# Plotting the evoked response recorded be a single channel having index idx
def plot_channel_evoked(evoked, idx, ch_type, c='k'):
    
    times = evoked.times
    evoked_data = evoked.get_data()
    ch_name = evoked.info['ch_names'][idx]
    
    sig = evoked_data[idx, :].flatten()
    sig, _ = simulations.max_scaling(sig, sig)
    
    fig = plt.figure(figsize=(10, 4))
    gs_fig = gridspec.GridSpec(1, 2, figure=fig, wspace=0.05, width_ratios=[4, 1])
        
    ax_sig = fig.add_subplot(gs_fig[0])
    ax_topo = fig.add_subplot(gs_fig[1])
    ax_sig.plot(times, sig, c=c, label=ch_name)
    #ax_sig.set_title(ch_name)
    ax_sig.tick_params(axis='both', labelsize=9) 
    ax_sig.set_ylabel('amplitude (a.u.)', fontsize=10)
    ax_sig.set_xlabel('time (s)', fontsize=10)
    [ax_sig.spines[side].set_visible(False) for side in ['right', 'top']]
    ax_sig.axhline(y=0, linewidth=1, linestyle='dashed', color='grey', alpha=0.5)
    ax_sig.grid(axis = 'x', linestyle = '--')
    ax_sig.legend()
    
    _, plot_locs = ch_locs.get_ch_locs_2d(evoked.info, ch_type)
    ch_names = evoked.info['ch_names']
    comp_loc =  plot_locs[ch_names[idx]]
    locs = np.array([plot_locs[ch] for ch in ch_names])
    ax_topo.scatter(locs[:,0], locs[:,1], c='lightgrey', s=10)
    ax_topo.scatter(comp_loc[0], comp_loc[1], c='r', s=15)
    ax_topo.axis('off')
    ax_topo.set_aspect('equal') 
    
def plot_results(results, lower, est_ylabels, colors, save_path=None, title=None, p_lim=0.01):

    # Forming the outer grid 
    fig = plt.figure(figsize=(25, 5))
    if title:
        fig.suptitle(title)
    outer = gridspec.GridSpec(1, 4)
    grids = np.arange(4)
    
    for i, est, c, y_lab in zip(grids, results, colors, est_ylabels):
        
        # Values to plot
        result = results[est]
        lower_values = lower[est]
        
        # Forming the inner grid
        inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[i], width_ratios=[4, 1], wspace=0.01)
        ax_scatter = fig.add_subplot(inner[0])
        ax_hist = fig.add_subplot(inner[1], sharey=ax_scatter)
        
        # Scatter plot
        ax_scatter.scatter(np.arange(len(result)), result, color=c, s=5, alpha=0.8)
        lower_med = np.median(lower_values)
        lower_max = np.max(lower_values)
        lower_min = np.min(lower_values)
        ax_scatter.axhline(y=lower_med, linewidth=1, linestyle='dashed', color='grey', alpha=0.5)
        ax_scatter.fill_between([0, len(result)], lower_max, lower_min,color='grey', alpha=0.2)
        ax_scatter.set_ylim([-0.1, 1.1])
        ax_scatter.set_title(est)
        ax_scatter.tick_params(axis='both', labelsize=10)
        ax_scatter.set_xlabel('channel index')
        ax_scatter.set_ylabel(y_lab)
        ax_scatter.grid(which='both', linestyle='--', alpha=0.5)
        
        # p-value line
        p_per = 100 - p_lim*100
        p_lim = np.percentile(lower_values, p_per)
        ax_scatter.axhline(y=p_lim, linewidth=1, linestyle='dashed', color='k', alpha=0.5, label=f'p={p_lim}')

        # Histogram
        ax_hist.hist(lower_values, bins=30, orientation='horizontal', color=c, edgecolor='none', alpha=0.7)
        ax_hist.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        for spine in ['top', 'right', 'bottom', 'left']:
            ax_hist.spines[spine].set_visible(False)
    
    plt.subplots_adjust(hspace=0.33, wspace=0.15)
    plt.show()
    
def plot_topo_order(est_res, info, cmap, ax, est, ch_type, comp_name=None, chs_higlight=None):
    
    _, plot_locs = ch_locs.get_ch_locs_2d(info, ch_type=ch_type)
    _, sorted_names, sorted_values = test_est.order_magnitude(est_res, info, ch_type)
    
    zero_mask = np.isclose(sorted_values, 0, atol=1e-8)
    if np.any(zero_mask):
        num_zeros = np.sum(zero_mask)
        not_signi = sorted_names[:num_zeros]
        not_sig_locs = [plot_locs[ch] for ch in not_signi]
        for loc in not_sig_locs:
            ax.scatter(loc[0], loc[1], 
                       facecolors='none', edgecolors='black',
                       linewidths=0.5, alpha=0.6, s=20)
        signi = sorted_names[num_zeros:]
    else:
        signi = sorted_names
        
    colors = [cmap(i / (len(signi) - 1)) for i in range(len(signi))] 
    locs = [plot_locs[ch] for ch in signi]
    
    for loc, col in zip(locs, colors):
        ax.scatter(loc[0], loc[1], color=col, s=20)
        
        if comp_name:
            comp_loc = plot_locs[comp_name]
            ax.scatter(comp_loc[0], comp_loc[1], color='r', s=25)
                       
        ax.axis('off')
    
    if chs_higlight:
        for ch_name, marker in chs_higlight:
            ch_loc = plot_locs[ch_name]
            ax.scatter(ch_loc[0], ch_loc[1], 
                       facecolors='none', edgecolors='r',
                       marker=marker, s=60, linewidths=0.6, alpha=0.5)
    ax.set_title(est, fontsize=12)


def plot_topo_orders(results, est_list, ch_type, comp_chan, evoked_comp):
    
    # Setting the axis
    n_est = len(est_list)
    fig, axs = plt.subplots(1, n_est, figsize=(12,2))
    axes = axs.flatten()

    # Plotting the ordered topos
    for ax, est in zip(axes, est_list):
        plot_topo_order(results[est], evoked_comp.info, 
                        cmap=matplotlib.colormaps['Greys'], ax=ax, 
                        est=est, ch_type=ch_type, comp_name=comp_chan)


# Plotting 2D topographic map of the signal strenght in given time windows
def plot_strenght_topos(evoked, ch_type, time_windows, av_type='abs', ch_locs_highlight=None, window_titles=None, cmap='Greys'):
    
    # Getting the positions of the different channels
    pos, pos_dict = ch_locs.get_ch_locs_2d(evoked.info, ch_type)

    # Titles to use with the windows if one is not already given
    if window_titles is None:
        window_titles = [f'{np.round(start,2)} - {np.round(end, 2)}' for start, end in time_windows]

    # Setting up the axis for plotting
    n_windows = len(time_windows)
    fig, axes = plt.subplots(1, n_windows, figsize=(12, 3))

    # Looping through the time windows
    for i, window in enumerate(time_windows):

        if n_windows == 1:
            ax = axes
        else:
            ax = axes[i]
        # Plotting the map in the given window
        tmin = window[0]
        tmax = window[1]
        i_min = evoked.time_as_index(tmin)[0]
        i_max = evoked.time_as_index(tmax)[0]
        data_window = evoked.data[:, i_min:i_max+1]
        data_avg = data_window.mean(axis=1)
        # TO DO: I still want to check whether I want to keep all of these 
        if av_type == 'abs':
            data_plot = np.abs(data_avg)
        if av_type == 'rms':
            data_plot = np.sqrt((data_avg ** 2))
        if av_type == 'sqr':
            data_plot = data_avg ** 2

        im, _ = mne.viz.plot_topomap(
            data=data_plot,
            pos=pos,
            axes=ax,
            show=False,
            sensors=True, 
            contours=0,
            sphere=0.1,
            cmap=cmap
        )

        # Highlighting the location of possibly given channels
        if ch_locs_highlight is not None: 
            for ch_name, marker, color in ch_locs_highlight:
                highlight_pos = pos_dict[ch_name]
                ax.scatter(
                    highlight_pos[0],
                    highlight_pos[1],
                    s=20,
                    facecolors='none', 
                    edgecolors=color, 
                    marker=marker,
                    linewidths=1)
        # Adding the title
        ax.set_title(window_titles[i])

    plt.show()

def plot_sig_comp(evoked, results, p_vals, ch_type,
                  comp_chan, channel2,
                  c1='black', c2='darkgrey'): 
    
    idx1 = mne.pick_channels(evoked.info['ch_names'], include=[comp_chan])[0]
    idx2 = mne.pick_channels(evoked.info['ch_names'], include=[channel2])[0]
    
    times = evoked.times
    evoked_data = evoked.get_data()
    
    sig1 = evoked_data[idx1, :].flatten()
    sig2 = evoked_data[idx2, :].flatten()
    sig1, sig2 = simulations.max_scaling(sig1, sig2)
    
    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])

    # Main axis (big one on the left)
    ax_series = fig.add_subplot(gs[0])

    # Create a sub-gridspec inside the second column
    gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1])

    # Now split the second axis vertically into two
    ax1a = fig.add_subplot(gs_right[0])
    ax1b = fig.add_subplot(gs_right[1])

    # Plotting the signals
    ax_series.plot(times, sig1, label=comp_chan, c=c1, linewidth=1)
    ax_series.plot(times, sig2, label=channel2, c=c2, linewidth=1)
    ax_series.set_ylabel('amplitude (a.u.)', fontsize=10)
    ax_series.set_xlabel('time (s)', fontsize=10)
    ax_series.tick_params(axis='both', labelsize=10) 
    [ax_series.spines[side].set_visible(False) for side in ['right', 'top']]
    ax_series.grid(axis = 'x', linestyle = '--')
    ax_series.axhline(y=0, linewidth=1, linestyle='dashed', color='grey', alpha=0.5)
    ax_series.set_ylim([-1.1,1.1])
    ax_series.legend(loc='lower center', bbox_to_anchor=(0.5, -0.85), ncol=2)
    
    # Adding the sensor location
    locs, plot_locs = ch_locs.get_ch_locs_2d(evoked.info, ch_type=ch_type)
    ax1a.scatter(locs[:,0], locs[:,1], c='lightgrey', s=6)
    loc1 =  plot_locs[comp_chan]
    ax1a.scatter(loc1[0], loc1[1], facecolors='none', edgecolors='k', marker='.', s=30)
    loc2 =  plot_locs[channel2]
    ax1a.scatter(loc2[0], loc2[1], facecolors='none', edgecolors='k', marker='.', s=30)
    ax1a.axis('off')
    ax1a.set_aspect('equal')

    # Adding value information with a text box
    pcc_mi = np.round(results['PCC'][idx2], 2)
    pcc_p = p_vals['PCC'][idx2]
    ksg_mi = np.round(results['KSG'][idx2], 2)
    ksg_p = p_vals['KSG'][idx2]
    kde_mi = np.round(results['KDE'][idx2], 2)
    kde_p = p_vals['KDE'][idx2]
    ab_mi = np.round(results['AB'][idx2], 2)
    ab_p = p_vals['AB'][idx2]
    info = f'PCC={pcc_mi} (p={pcc_p})\nKDE={kde_mi} (p={kde_p})\nKSG={ksg_mi} (p={ksg_p})\nAB={ab_mi} (p={ab_p})'
    ax1b.text(0.5, 0.4, info,
             ha='center', va='center',
             fontsize=12,
             bbox=dict(boxstyle='round', facecolor='whitesmoke', edgecolor='gray'),
             transform=ax1b.transAxes)
    ax1b.axis('off')
    

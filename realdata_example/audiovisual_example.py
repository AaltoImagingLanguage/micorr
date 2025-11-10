
# %% imports

# general imports
import mne
from functools import partial

# imports from our library
from micorr.estimators import mi_estimators, corr_est
from micorr.realdata import test_est, plotting

# %% Loading the mne sample data

sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = (sample_data_folder / 'MEG' /
                        'sample' / 'sample_audvis-ave.fif')
                        
# Here, we are choosing to use left auditory stimuli from the data (this can be changed)
condition = 'Left Auditory'
evoked = mne.read_evokeds(sample_data_raw_file,
                          condition=condition, # Picking the condition
                          baseline=(-0.2, 0) # Adding baseline correction
                          )

# %% Choosing the channel type & getting the data

ch_type = 'planar2'  # This can be 'mag', 'planar 1', 'planar 2' or 'all'

if ch_type != 'all':
    evoked_comp = evoked.copy().pick_types(meg=ch_type)
    evoked_comp_data = evoked_comp.get_data()
else:
    evoked_comp_data = evoked.get_data()

# %% Plotting the evoked responses

evoked_comp.plot_topo()

# %% Choosing the channel to compare

comp_chan = 'MEG 2423'
idx_comp = mne.pick_channels(
    evoked_comp.info['ch_names'], include=[comp_chan])[0]
sig_comp = evoked_comp_data[idx_comp, :].flatten()
plotting.plot_channel_evoked(evoked_comp, idx_comp, ch_type=ch_type)

# %% Setting up the parameters the estimators

snr = 30  # Noise to be added

# Estimators to be tested
est_list = {'PCC': corr_est.abs_correlation,
            'KDE': mi_estimators.kde_mi,
            'KSG': partial(mi_estimators.mi_ksg, k=5),
            'AB': mi_estimators.binned_mi}

# How many runs are used in estimating the lower bound
lowerbound_runs = 1_000

# %% Applying the estimators

print('Applying the estimators')
results = test_est.cal_rel_values(
    est_list, sig_comp, evoked_comp_data, snr=snr)

print('Defining the lower bound')
random_lower = test_est.lowerbound_random(
    est_list, sig_comp, n_runs=lowerbound_runs)

print('Calculating the p-values')
p_vals = test_est.calc_pvals(results, random_lower)

# %% PLotting the results

# Paremeter for plotting
est_ylabels = ['$|r|$', 'NMI',
               'NMI', 'NMI']
colors = ['#72E059', '#FDD466', '#6680E3', '#E87C6E']

# Plotting the results
plotting.plot_results(results, random_lower, est_ylabels, colors)

# %% Plotting the rank ordering

plotting.plot_topo_orders(results, est_list, ch_type, comp_chan, evoked_comp)

#%% Plotting signal strenght (at different time windows) for comparision

# Time windows we want to plot
time_windows = [[0.05, 0.100], [0.100, 0.15]]

# Now we are just highlighting the defined comparision channel 
ch_locs_highlight = [(comp_chan, '.', 'r')]

plotting.plot_strenght_topos(evoked_comp, ch_type, time_windows, ch_locs_highlight=ch_locs_highlight)

#%% Plotting an example case 

# Choosing the comparision channel to plot
comp_chan2 = 'MEG 0243' 
plotting.plot_sig_comp(evoked_comp, results, p_vals, ch_type,
                       comp_chan=comp_chan,
                       channel2=comp_chan2) 

#%%

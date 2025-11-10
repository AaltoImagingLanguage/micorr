
import mne
import numpy as np
from mne.channels.layout import _find_topomap_coords

def get_ch_locs_2d(info, ch_type):

    '''
    Getting the 2D locations of the MEG channels with the given type.
    
    Parameters
    ----------
    info : mne.Info
        The mesurement info associated with the M/EEG data.
    ch_type : str
        Channel type of interest.
        Options: 'mag', 'planar1' or 'planar2'
        
    Returns
    -------
    pos : np.ndarray
        Positions of channels in 2D.
    pos_dict: dict
    	Dictionary with channel names and corresponding positions in 2D.
    '''
    
    picks_meg = mne.pick_types(info, meg=True)
    meg3d_posistions = [tuple(np.round(info['chs'][i]['loc'][:3], 5)) for i in picks_meg]
    unique_3dpos = list(set(meg3d_posistions))
    groups = {pos:[] for pos in unique_3dpos}
    meg_ch_names = [info['ch_names'][i] for i in picks_meg]
    
    for pos_3d, ch in zip(meg3d_posistions, meg_ch_names):
        groups[pos_3d].append(ch)
        
    if ch_type == 'mag':
        picks_mag = mne.pick_types(info, meg='mag')
        ch_names_mag = [info['ch_names'][i] for i in picks_mag]
        pos = _find_topomap_coords(info, picks=picks_mag)
        pos_dict = {ch_names_mag[i]: pos[i] for i in range(len(pos))}
    
    if ch_type == 'planar1':
        picks_pla1 = mne.pick_types(info, meg='planar1')
        ch_names_pla1 = [info['ch_names'][i] for i in picks_pla1]
        pos = _find_topomap_coords(info, picks=picks_pla1)
        pos_dict = {ch_names_pla1[i]: pos[i] for i in range(len(pos))}
    
    if ch_type == 'planar2':
        picks_pla2 = mne.pick_types(info, meg='planar2')
        ch_names_pla2 = [info['ch_names'][i] for i in picks_pla2]
        pos = _find_topomap_coords(info, picks=picks_pla2)
        pos_dict = {ch_names_pla2[i]: pos[i] for i in range(len(pos))}
        
    return pos, pos_dict

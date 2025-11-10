
import numpy as np 
import matplotlib.pyplot as plt
from . import transformations
from sklearn.preprocessing import StandardScaler


#TO FIX: Should I add some defaults in here we could add our default response form
def simulate_evoked(start, end, fs, peak_time, peak_duration, sin_shift, plotting=False):
    
    # Simulating time array
    times = np.arange(start, end, 1/fs)
    freq_sin = 1/(peak_duration)
    std_gaus = 0.36*peak_duration
    
    # TO FIX: This one could still be checked
    if sin_shift > std_gaus*0.7:
        raise ValueError('Too high value provided for sin_shift')
        
    # Aligning a positive peak of the sinewave with the Gaussian and shifting this a little to obtain the skewness
    shift_align = (np.pi / 2 - 2 * np.pi * freq_sin * peak_time)/(2 * np.pi * freq_sin) + sin_shift
    # Simulating the sine wave
    sine_wave = np.sin(2 * np.pi * freq_sin * (times + shift_align))
    # Simulting the Gaussian
    gauss = np.exp(-((times-peak_time)**2/(2*std_gaus**2)))
    # Forming the Morlet wavelet
    evoked = gauss*sine_wave
    
    # Ensuring the the peaks aling (because we shifted sinewave we need to correct the damage done with that)
    if sin_shift != 0:
        peak_index = np.where(np.abs(evoked) == np.max(np.abs(evoked)))[0]
        zero_index = np.where(np.isclose(times, peak_time, atol=10E-6) == True)[0]
        samp_shift = -(peak_index - zero_index)
        _ , evoked = transformations.time_shift(evoked, samp_shift)

    # An optional plotting the simulation process
    if plotting == True:
        fig, ax = plt.subplots(2,1, figsize = (10,10))
        ax[0].plot(times, gauss, label = 'Gaussian (envelope)', c = 'b')
        ax[0].set_xlabel('time(s)', fontsize = 20)
        ax[0].set_ylabel('amplitude (a.u.)', fontsize = 20)
        ax[0].set_title('functions', fontsize = 20, y = 1)
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].plot(times, sine_wave, label = 'Sine wave (carrier)', c = 'r')
        ax[0].legend(fontsize = 15, loc = 1)
        ax[0].tick_params(axis='both', which='major', labelsize=15)
        ax[0].tick_params(axis='both', which='major', labelsize=15)
        ax[1].plot(times,evoked, c = 'k')
        ax[1].set_xlabel('time(s)', fontsize = 20)
        ax[1].set_ylabel('amplitude (a.u.)', fontsize = 20)
        ax[1].set_title('product', fontsize = 20)
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].tick_params(axis='both', which='major', labelsize=15)
        fig.subplots_adjust(hspace = 0.4)
        plt.show()
        
    return evoked, times

def max_scaling(signal1, signal2):
    
    '''
    Scales the given signals based on the shared maximum absolute amplitude.
    
    Parameters
    ----------
    signal1: np.ndarray
        The first signal to scale
    signal2: np.ndarray, optional
        The second signal to scale.
        
    Returns
    -------
    s_sig1: np.ndarray
        Scaled first signal
    s_sig2: np.ndarray
        Scaled second signal
        
    '''
        
    norm_factor = np.max([np.max(np.abs(signal1)), np.max(np.abs(signal2))])
    # Avoiding the division by zero
    if norm_factor == 0:
        raise ValueError('The provided signal contains only zeros and cannot be scaled.')
    else:
        scaled_sig1 = signal1/norm_factor
        scaled_sig2 = signal2/norm_factor
        
    return scaled_sig1, scaled_sig2
    
def std_norm(signal1, signal2):
   scaler = StandardScaler()
   normalized_signal1 = scaler.fit_transform(signal1.reshape(-1, 1)).flatten()
   normalized_signal2 = scaler.fit_transform(signal2.reshape(-1, 1)).flatten()
   return normalized_signal1, normalized_signal2
  
#--------- functions below will probably be removed ------------  

# TO FIX: Add description in here
# I feel like this one is a bit extra and I am noot sure whether to puplish it or not 
def simulate_waves(start, end, samp_freq, amplitude_sin, freq_sin, phi = 0, plotting = False):
 
    times = np.arange(start, end, 1/samp_freq)
    sine_wave = amplitude_sin*np.sin(2*np.pi*freq_sin*times + phi)

    if plotting == True:
        fig, ax = plt.subplots(1,1, figsize = (10,10))
        ax.plot(times, sine_wave, label = 'Sine wave', c = 'b')
        ax.set_xlabel('time(s)', fontsize = 20)
        ax.set_ylabel('amplitude (\u03bcV/fT)', fontsize = 20)
        ax.set_title('Sine wave, f = {f} Hz, phi = {phi} rad'.format(
            f=freq_sin, phi=phi), fontsize = 20, y = 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize = 15, loc = 1)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.tick_params(axis='both', which='major', labelsize=15)
        plt.show()
        
    return sine_wave, times
 
# TO FIX: I would maybe like to keep this function 
# Same with this one not sure if I am going to publish this one 
def simulate_conevoked(start, end, samp_freq, height_gaus, center_gaus, std_gaus, amplitude_sin, freq_sin, rep):
    """
    Generates the signal with repeated evoked responses.
    
    Parameters:
    start (float): The time (in seconds) at which the first response begins.
    end (float):  The time (in seconds) at which the first response ends.
    samp_freq (float): The sampling frequency of the generated response.
    height_gaus (float): The peak height of the Gaussian function.
    center_gaus (float): The position of the center of the Gaussian function.
    std_gaus (float): The standard deviation of the Gaussian function.
    amplitude_sin (float): The amplitude of the sinusoidal function. 
    freq_sin (float): The frequency of the sinusoidal function.
    rep (int): The number of times the response is repeated.
        
    Returns:
    np.ndarray: The generated signal with the repeated responses.
    """
    
    # This line does not correspond with the updated simulate_evoked function
    evoked, _ = simulate_evoked(start, end, samp_freq, height_gaus, center_gaus, std_gaus, amplitude_sin, freq_sin)
    new_evoked = np.concatenate([evoked,evoked])
    for i in range(rep-2):
        new_evoked = np.concatenate([new_evoked,evoked])
    new_times = np.arange(start, rep * (end-start) + start, 1/samp_freq)
    return new_evoked, new_times



    

import numpy as np 
import math 

def calculate_SNR(signal, noise):
    return 10 * math.log10(np.mean(np.square(signal))/np.mean(np.square(noise)))

def snr_to_noise(snr_aim, signal, pres=0.5):

    noise_std = snr_to_std(snr_aim, signal)
    noise = np.random.normal(0, noise_std, size=len(signal))
    snr_true = calculate_SNR(signal, noise)

    while snr_true < snr_aim-pres or snr_true > snr_aim+pres:
            noise = np.random.normal(0, noise_std, size=len(signal))
            snr_true = calculate_SNR(signal, noise)

    return noise, noise_std

def snr_to_std(snr_aim, signal):
    signal_power = np.mean(np.square(signal))
    snr_linear =  10**(snr_aim/10)
    noise_power = signal_power/snr_linear
    noise_std = np.sqrt(noise_power)
    return noise_std
    


# micorr

## Description

This project provides a framework that can be used to test different pairwise similarity estimators on simulated waveforms. 
The intended use of the code provided here, is to compare common mutual information, including Kraskow-Stögbauer-Grassberger (KSG), kernel density estimator (KDE) and adaptive binning based estimator with Pearson correlation on simulated even-related brain responses.
In our example cases, the simulated brain resposese are transformed with transformations that include changes in the noise level, sample size, linear scaling and other non-linear transformations.
Users can easily edit the code provided by adding their own estimators, tranformations or base waveforms in the simulations. This feature makes the provided code suitable for comparing different similarity estimators in more varied scenarios.

Further, we provide a framework for testing the estimators with real M/EEG data. Including real example case study with the MNE sample dataset
openly available from the MNE-Python project:
https://mne.tools/stable/overview/datasets_index.html#sample.

## Depedencies

This project depends on the following Python libraries:

- numpy (1.26.4)
- matplotlib (3.10.1)
- scipy (1.14.0)
- ennemi (1.3.0)
- mne (1.7.1)
- scikit-learn (1.5.0)
- tqdm (4.66.4)

See requirements.txt for a full list of dependencies.

## Usage

Examples are divided into three subfolders.

See [`transformations_examples`](./transformations_examples) folder for demonstrations of various simulation cases, including:
- SNR
- Sample size
- Outliers
- Noisy segments
- Time-shift
- Duration differences
- Quadric relationship
- Amplitude (scaling) differences

See [`freeparameter_testing`](./freeparameter_testing) folder for examples of the impact of the freeparameter choices with MI estimators (AB, KSG and KDE).

See [`realdata_example`](./realdata_example) folder for example testing the estimator with real MEG data (MNE sample dataset).

## References

This project makes use of the following works and resources:

- Gramfort A. et al. (2013). *MEG and EEG data analysis with MNE-Python.* Frontiers in Neuroscience, 7:267. 
- Laarne, P. et al. (2021). *ennemi: Non-linear correlation detection
with mutual information.* SoftwareX, 1
- Virtanen, P. et al. (2020). *SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python.*
Nature Methods, 17, 261–272.
- Moon, Y.-I. et al. (1995). *Estimation of mutual information using
kernel density estimators.* Physical Review E, 52, 2318–2321.
- Kraskov, A. et al. (2004). *Estimating mutual information.*
Physical Review E, 69, 066138.

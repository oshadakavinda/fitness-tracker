import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 3                                               #
#                                                            #
##############################################################

# Updated by Dave Ebbelaar on 22-12-2022

from sklearn.decomposition import PCA
from scipy.signal import butter, lfilter, filtfilt
import copy


# This class removes the high frequency data (that might be considered noise) from the data.
# We can only apply this when we do not have missing values (i.e. NaN).
class LowPassFilter:
    def low_pass_filter(
        self,
        data_table,
        col,
        sampling_frequency,
        cutoff_frequency,
        order=5,
        phase_shift=True,
    ):
        # http://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
        # Cutoff frequencies are expressed as the fraction of the Nyquist frequency, which is half the sampling frequency
        nyq = 0.5 * sampling_frequency
        cut = cutoff_frequency / nyq

        b, a = butter(order, cut, btype="low", output="ba", analog=False)
        if phase_shift:
            data_table[col + "_lowpass"] = filtfilt(b, a, data_table[col])
        else:
            data_table[col + "_lowpass"] = lfilter(b, a, data_table[col])
        return data_table


# Class for Principal Component Analysis. We can only apply this when we do not have missing values (i.e. NaN).
# For this we have to impute these first, be aware of this.
class PrincipalComponentAnalysis:

    pca = []

    def __init__(self):
        self.pca = None

    def normalize_dataset(self, data_table, columns):
        dt_norm = copy.deepcopy(data_table)
        for col in columns:
            dt_norm[col] = (data_table[col] - data_table[col].mean()) / (
                data_table[col].max()
                - data_table[col].min()
                # data_table[col].std()
            )
        return dt_norm

    # Perform the PCA on the selected columns and return the explained variance.
    def determine_pc_explained_variance(self, data_table, cols):

        # Normalize the data first.
        dt_norm = self.normalize_dataset(data_table, cols)

        # perform the PCA.
        self.pca = PCA(n_components=len(cols))
        self.pca.fit(dt_norm[cols])
        # And return the explained variances.
        return self.pca.explained_variance_ratio_

    # Apply a PCA given the number of components we have selected.
    # We add new pca columns.
    def apply_pca(self, data_table, cols, number_comp):

        # Normalize the data first.
        dt_norm = self.normalize_dataset(data_table, cols)

        # perform the PCA.
        self.pca = PCA(n_components=number_comp)
        self.pca.fit(dt_norm[cols])

        # Transform our old values.
        new_values = self.pca.transform(dt_norm[cols])

        # And add the new ones:
        for comp in range(0, number_comp):
            data_table["pca_" + str(comp + 1)] = new_values[:, comp]

        return data_table


import os
print(os.getcwd())  # Prints the current working directory

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("os.getcwd()+../../data/interim/02_outliers_removed_chauvenets.pkl")

sensor_col = list(df.columns[:6])


plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20, 5)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['lines.linewidth'] = 2
# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

for col in sensor_col:
    df[col] = df[col].interpolate()

df.info()

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

#We know that: the heavy set contains 5 repetitions, and medium set contains 10 repetitions for each exercise

#Now we need to know the duration for each set

for set in  df['set'].unique():
    
    strart = df[df['set'] == set].index[0]
    end = df[df['set'] == set].index[-1]
    
    duration = end - strart
    
    df.loc[(df['set'] == set) , 'duration'] = duration.seconds
    

duration_df =  df.groupby('category')['duration'].mean()    

duration_df[0] / 5 # so each repetition take 2.9 sec in heavy set
duration_df[1] / 10 # so each repetition take 2.4 sec in medium set


# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()

LowPass = LowPassFilter()

sampling_frq = 1000 / 200 # # because we are taking the record every 200 ms, so that line calculates number of repetition in 1 sec

cutoff_frq = 1.3 # the low cutoff frequency, meaning more smoothing in signal

LowPass.low_pass_filter(df_lowpass , 'acc_y' , sampling_frq , cutoff_frq)

subset = df_lowpass[df_lowpass["set"] == 45]
print(subset["label"][0])


fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="raw data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="butterworth filter")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
plt.show()


for col in sensor_col:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, sampling_frq, cutoff_frq, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = df_lowpass.copy()


pca_instance = PrincipalComponentAnalysis()
pc_values = pca_instance.determine_pc_explained_variance(df_pca, sensor_col)

plt.figure(figsize=(10,10))
plt.plot(range(1, 7) , pc_values)
plt.show()

df_pca = pca_instance.apply_pca(df_pca , sensor_col , 3 )
subset = df_pca[df_pca["set"] == 35] 

subset[['pca_1' , 'pca_2' , 'pca_3']].plot()

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

# To further exploit the data, the scalar magnitudes r of the accelerometer and gyroscope were calculated. 
# r is the scalar magnitude of the three combined data points: x, y, and z. 
# The advantage of using r versus any particular data direction is that it is impartial to device orientation and can handle dynamic re-orientations.
# r is calculated by: r_{magnitude} = sqrt{x^2 + y^2 + z^2}

df_squares = df_pca.copy()

acc_r = df_squares['acc_x'] **2 + df_squares['acc_y'] **2 +df_squares['acc_z'] **2 
gyr_r = df_squares['gyr_x'] **2 + df_squares['gyr_y'] **2 +df_squares['gyr_z'] **2 


df_squares['acc_r'] = np.sqrt(acc_r)
df_squares['gyr_r'] = np.sqrt(gyr_r)



subset = df_squares[df_pca["set"] == 18] 

subset[['acc_r' , 'gyr_r' ]].plot(subplots = True)

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------



# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
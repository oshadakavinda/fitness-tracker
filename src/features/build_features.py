import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




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


class NumericalAbstraction:

    # This function aggregates a list of values using the specified aggregation
    # function (which can be 'mean', 'max', 'min', 'median', 'std')
    def aggregate_value(self, aggregation_function):
        # Compute the values and return the result.
        if aggregation_function == "mean":
            return np.mean
        elif aggregation_function == "max":
            return np.max
        elif aggregation_function == "min":
            return np.min
        elif aggregation_function == "median":
            return np.median
        elif aggregation_function == "std":
            return np.std
        else:
            return np.nan

    # Abstract numerical columns specified given a window size (i.e. the number of time points from
    # the past considered) and an aggregation function.
    def abstract_numerical(self, data_table, cols, window_size, aggregation_function):

        # Create new columns for the temporal data, pass over the dataset and compute values
        for col in cols:
            data_table[
                col + "_temp_" + aggregation_function + "_ws_" + str(window_size)
            ] = (
                data_table[col]
                .rolling(window_size)
                .apply(self.aggregate_value(aggregation_function))
            )

        return data_table


class FourierTransformation:

    # Find the amplitudes of the different frequencies using a fast fourier transformation. Here,
    # the sampling rate expresses the number of samples per second (i.e. Frequency is Hertz of the dataset).
    def find_fft_transformation(self, data, sampling_rate):
        # Create the transformation, this includes the amplitudes of both the real
        # and imaginary part.
        transformation = np.fft.rfft(data, len(data))
        return transformation.real, transformation.imag

    # Get frequencies over a certain window.
    def abstract_frequency(self, data_table, cols, window_size, sampling_rate):

        # Create new columns for the frequency data.
        freqs = np.round((np.fft.rfftfreq(int(window_size)) * sampling_rate), 3)

        for col in cols:
            data_table[col + "_max_freq"] = np.nan
            data_table[col + "_freq_weighted"] = np.nan
            data_table[col + "_pse"] = np.nan
            for freq in freqs:
                data_table[
                    col + "_freq_" + str(freq) + "_Hz_ws_" + str(window_size)
                ] = np.nan

        # Pass over the dataset (we cannot compute it when we do not have enough history)
        # and compute the values.
        for i in range(window_size, len(data_table.index)):
            for col in cols:
                real_ampl, imag_ampl = self.find_fft_transformation(
                    data_table[col].iloc[
                        i - window_size : min(i + 1, len(data_table.index))
                    ],
                    sampling_rate,
                )
                # We only look at the real part in this implementation.
                for j in range(0, len(freqs)):
                    data_table.loc[
                        i, col + "_freq_" + str(freqs[j]) + "_Hz_ws_" + str(window_size)
                    ] = real_ampl[j]
                # And select the dominant frequency. We only consider the positive frequencies for now.

                data_table.loc[i, col + "_max_freq"] = freqs[
                    np.argmax(real_ampl[0 : len(real_ampl)])
                ]
                data_table.loc[i, col + "_freq_weighted"] = float(
                    np.sum(freqs * real_ampl)
                ) / np.sum(real_ampl)
                PSD = np.divide(np.square(real_ampl), float(len(real_ampl)))
                PSD_pdf = np.divide(PSD, np.sum(PSD))
                data_table.loc[i, col + "_pse"] = -np.sum(np.log(PSD_pdf) * PSD_pdf)

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

df[df["set"]==25]["acc_y"].plot()
df[df["set"]==50]["acc_y"].plot()

plt.show()



for set in  df['set'].unique():
    
    start = df[df['set'] == set].index[0]
    end = df[df['set'] == set].index[-1]
    
    duration = end - start
    
    df.loc[(df['set'] == set) , 'duration'] = duration.seconds
    


duration_df =  df.groupby('category')['duration'].mean()    

duration_df[0] / 5 # so each repetition take 2.9 sec in heavy set
duration_df[1] / 10 # so each repetition take 2.4 sec in medium set


# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------


# Create a copy of the dataframe
df_lowpass = df.copy()

# Initialize LowPass filter
LowPass = LowPassFilter()

# Low-pass filter parameters
sampling_frq = 1000 / 200  # sampling frequency (5 Hz)
cutoff_frq = 1.3  # cutoff frequency

# df_lowpass = LowPass.low_pass_filter(df_lowpass , 'acc_y' , sampling_frq , cutoff_frq,order=5)
# subset = df_lowpass[df_lowpass["set"] == 45]
# print(subset["label"][0])

# fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
# ax[0].plot(subset["acc_y"].reset_index(drop=True), label="raw data")
# ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="butterworth filter")
# ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
# ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
# plt.show()
pd.set_option('display.max_columns', None)
# Create a copy of the original dataframe for raw data
df_raw = df.copy()

# Create a copy for low-pass filtered data
df_lowpass = df.copy()

# Apply low-pass filter to all sensor columns
for col in sensor_col:
    # Apply low-pass filter
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, sampling_frq, cutoff_frq, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]

# Select a specific set (e.g., set 45)
subset_raw = df_raw[df_raw["set"] == 45]
subset_lowpass = df_lowpass[df_lowpass["set"] == 45]

# Create subplots for each sensor column
fig, axs = plt.subplots(len(sensor_col), 2, figsize=(20, 4*len(sensor_col)), sharex=True)
fig.suptitle(f'Raw vs Low-Pass Filtered Sensor Data for Set {subset_raw["set"].iloc[0]}', fontsize=16)

# Plot data for each sensor column
for i, col in enumerate(sensor_col):
    # Raw data plot
    axs[i, 0].plot(subset_raw[col].reset_index(drop=True), label='Raw')
    axs[i, 0].set_title(f'Raw {col} Data')
    axs[i, 0].set_ylabel(col)
    axs[i, 0].legend()

    # Low-pass filtered data plot
    axs[i, 1].plot(subset_lowpass[col].reset_index(drop=True), label='Low-Pass', color='red')
    axs[i, 1].set_title(f'Low-Pass Filtered {col} Data')
    axs[i, 1].set_ylabel(col)
    axs[i, 1].legend()

# Adjust layout and show plot
plt.tight_layout()
plt.show()


# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = df_lowpass.copy()


pca_instance = PrincipalComponentAnalysis()
pc_values = pca_instance.determine_pc_explained_variance(df_pca, sensor_col)

plt.figure(figsize=(10,10))
plt.plot(range(1, 7) , pc_values)
plt.xlabel("principal componeent number")
plt.ylabel("explained variance")
plt.show()

df_pca = pca_instance.apply_pca(df_pca , sensor_col , 3 )
# subset = df_pca[df_pca["set"] == 35] 

# subset[['pca_1' , 'pca_2' , 'pca_3']].plot()
# plt.show()

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

# To further exploit the data, the scalar magnitudes r of the accelerometer and gyroscope were calculated. 
# r is the scalar magnitude of the three combined data points: x, y, and z. 
# The advantage of using r versus any particular data direction is that it is impartial to device orientation and can handle dynamic re-orientations.
# r is calculated by: r_{magnitude} = sqrt{x^2 + y^2 + z^2}



# Assuming you've already created df_squares with the calculations you mentioned
df_squares = df_pca.copy()
acc_r = df_squares['acc_x']**2 + df_squares['acc_y']**2 + df_squares['acc_z']**2
gyr_r = df_squares['gyr_x']**2 + df_squares['gyr_y']**2 + df_squares['gyr_z']**2

df_squares['acc_r'] = np.sqrt(acc_r)
df_squares['gyr_r'] = np.sqrt(gyr_r)

# Select the subset for set 18
subset = df_squares[df_pca["set"] == 18]

# Create the plot
plt.figure(figsize=(12, 6))

# Plot acceleration resultant
plt.subplot(2, 1, 1)
plt.plot(subset['acc_r'])
plt.title('Acceleration Resultant Magnitude')
plt.xlabel('Sample')
plt.ylabel('Acceleration (m/s²)')

# Plot gyroscope resultant
plt.subplot(2, 1, 2)
plt.plot(subset['gyr_r'])
plt.title('Gyroscope Resultant Magnitude')
plt.xlabel('Sample')
plt.ylabel('Angular Velocity (rad/s)')

# Adjust layout and show plot
plt.tight_layout()
plt.show()

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal =  df_squares.copy() 
NumAbs = NumericalAbstraction()
sensor_col = sensor_col + ['acc_r' , 'gyr_r']
# NumAbs.abstract_numerical(df_temporal , sensor_col , window_size=5 ,aggregation_function= 'mean' )
# we need to make moving average on each set because each set may containing different label (exercise)
 
df_temporal_list = []
for set in df_temporal['set'].unique():
    subset = df_temporal[df_temporal['set'] == set].copy()
    
    for col in sensor_col:
        subset = NumAbs.abstract_numerical(subset , sensor_col , window_size=5 ,aggregation_function= 'mean' )
        subset = NumAbs.abstract_numerical(subset , sensor_col , window_size=5 ,aggregation_function= 'std' )

    df_temporal_list.append(subset)


df_temporal =  pd.concat(df_temporal_list)

df_temporal.info()

subset[['acc_y' , 'acc_y_temp_mean_ws_5' , 'acc_y_temp_std_ws_5']].plot()
subset[['gyr_y' , 'gyr_y_temp_mean_ws_5' , 'gyr_y_temp_std_ws_5']].plot()
plt.show()

# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

# The idea of a Fourier transformation is that any sequence of measurements we perform can be represented by a combination of sinusoid functionswith different frequencies 

#DFT can provide insight into patterns and trends that would not otherwise be visible. Additionally, the DFT can be used to reduce noise, allowing for more accurate models.
df_frq = df_temporal.copy().reset_index()
# df_frq
FreqAbs = FourierTransformation()

sampling_frq = int(1000 / 200)
window_size = int (2800 / 200)

df_frq = FreqAbs.abstract_frequency(df_frq , ['acc_y'] , window_size , sampling_frq)
subset = df_frq[df_frq['set'] == 15]
subset[['acc_y']].plot()
plt.show()
subset.columns
# Fourier transformation  abstracted the sign into its basic constituent elements

subset[['acc_y_max_freq',
        'acc_y_freq_weighted',
        'acc_y_pse',
       'acc_y_freq_0.0_Hz_ws_14',
       'acc_y_freq_0.357_Hz_ws_14',
       'acc_y_freq_0.714_Hz_ws_14',
       'acc_y_freq_1.071_Hz_ws_14']].plot()
plt.show()


df_freq_list = []
for set in df_frq['set'].unique():
    print(f'Applying Fourier transformation to set {set}')
    subset = df_frq[df_frq['set'] == set].reset_index(drop = True).copy()
    
    subset = FreqAbs.abstract_frequency(subset , sensor_col , window_size , sampling_frq)
    df_freq_list.append(subset)
    
df_frq =  pd.concat(df_freq_list).set_index('epoch (ms)' , drop=True)
df_frq = df_frq.drop('duration' , axis=1)

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

df_frq = df_frq.dropna()
df_frq = df_frq.iloc[: :2]

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

from sklearn.cluster import KMeans
df_cluster = df_frq.copy()

cluster_col = ['acc_x' , 'acc_y' , 'acc_z']
k_values = range(2,10)
inertias = []

for k in k_values:
    
    subset = df_cluster[cluster_col]
    kmeans = KMeans(n_clusters = k  , n_init=20 , random_state=0)
    label = kmeans.fit_predict(subset)
    
    inertias.append( kmeans.inertia_)
    
inertias
plt.plot(k_values , inertias , '--o' ) 
plt.xlabel("k")
plt.ylabel("Sum of squared values")

plt.show()
# So the 5 or 6 is the optimal number


kmeans = KMeans(n_clusters = 5  , n_init=20 , random_state=0)
subset = df_cluster[cluster_col]
df_cluster['cluster'] = kmeans.fit_predict(subset)
df_cluster

import joblib 

joblib.dump(kmeans , 'os.getcwd()+../../models/Clustering_model.pkl')

#plot clusters

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

# Plot Labels
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for l in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == l]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=l)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle('os.getcwd()+../../data/interim/03_data_features.pkl')
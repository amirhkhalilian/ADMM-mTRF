from memory_profiler import memory_usage
import time

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import resample
from sklearn.linear_model import Ridge, ElasticNet
import numpy as np

import naplib as nl
from naplib.visualization import strf_plot

from ADMM_mTRF.model import admm_mTRF

##############################################################################
# Example comparison between ADMM-mTRF and NapLib
# This example is adapted from the STRF example
# provided by Naplib here:
# https://naplib-python.readthedocs.io/en/latest/auto_examples/strf_fitting/plot_STRF_fitting_basics.html
#
# Written by Amir Khalilian, 2024

if __name__ == "__main__":
    # Set up the data, following NapLib
    # ---------------
    data = nl.io.load_speech_task_data()

    # This data contains the fields 'aud' and 'resp', which give the stimulus and neural responses
    print(f"aud stimulus shape for first trial : {data[0]['aud'].shape}")
    print(f"response shape for first trial : {data[0]['resp'].shape}")

    # first, we normalize the responses
    data['resp'] = nl.preprocessing.normalize(data=data, field='resp')

    # get auditory spectrogram for each stimulus sound
    data['spec'] = [nl.features.auditory_spectrogram(trial['sound'], 11025) for trial in data]

    # make sure the spectrogram is the exact same size as the responses
    data['spec'] = [resample(trial['spec'], trial['resp'].shape[0]) for trial in data]

    # Since the spectrogram is 128-channels, which is very large, we downsample it
    print(f"before resampling: {data['spec'][0].shape}")

    resample_kwargs = {'num': 32, 'axis': 1}
    data['spec_32'] = nl.array_ops.concat_apply(data['spec'], resample, function_kwargs=resample_kwargs)

    print(f"after resampling:  {data['spec_32'][0].shape}")

    # -------------------------------
    # Fit STRF Models with ElasticNet using NapLib
    # -------------------------------

    # set the model parameters
    tmin = 0 # receptive field begins at time=0
    tmax = 0.3 # receptive field ends at a lag of 0.4 seconds
    sfreq = 100 # sampling frequency of data

    # select number of channels to fit
    total_ch = 10
    X_train = data['spec_32']
    naplib_times = []
    admm_times = []
    naplib_mems = []
    admm_mems = []
    for n_ch in np.linspace(1,total_ch, total_ch):
        Y_train = [data[i]['resp'][:,0:int(n_ch)] for i in range(len(data))]

        # Fit STRF models using ElasticNet (L1 and L2 penalty) regression.
        # define the estimator to be used in this TRF model
        def fit_naplib():
            estimator = ElasticNet(l1_ratio=0.01)
            strf_naplib = nl.encoding.TRF(tmin, tmax, sfreq, estimator=estimator)
            strf_naplib.fit(X=X_train, y=Y_train)

        t0 = time.time()
        mem_naplib = memory_usage((fit_naplib,), interval=0.1, max_usage=True)
        t1 = time.time()
        naplib_time = t1 - t0
        print(f'number of channels: {n_ch}, naplib time:{naplib_time:2.2f}, mem: {mem_naplib:.2f} MiB')
        naplib_times.append(naplib_time)
        naplib_mems.append(mem_naplib)

        # -----------------------------------------------------
        # Fit STRF Models with ElasticNet -- ADMM implementaion
        # -----------------------------------------------------
        # define the ADMM-mTRF model
        def fit_admm():
            admm_mtrf = admm_mTRF(direction=1)
            admm_mtrf.train(X_train, Y_train,
                            sfreq, tmin, tmax,
                            alpha=1.0,
                            l1_ratio=0.01)
        t0 = time.time()
        mem_admm = memory_usage((fit_admm,), interval=0.1, max_usage=True)
        t1 = time.time()
        admm_time = t1 - t0
        print(f'number of channels: {n_ch}, admm time:{admm_time:2.2f}, mem:{mem_admm:.2f} MiB')
        admm_times.append(admm_time)
        admm_mems.append(mem_admm)


    # -------------------------
    # Plot fit time as a function of number of channels
    # -------------------------
    print(naplib_times)
    print(admm_times)

    # Apply Seaborn Style
    sns.set_theme(style="whitegrid")  # Seaborn style

    # Create Subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)

    # Plot timing results
    sns.lineplot(x=np.linspace(1,total_ch, total_ch),
                 y=naplib_times,
                 ax=axes[0],
                 color='purple',
                 label='NapLib',
                 linewidth=3,
                 marker='o',
                 markersize=8,
                 markeredgecolor='black', markeredgewidth=1.5)
    sns.lineplot(x=np.linspace(1,total_ch, total_ch),
                 y=admm_times,
                 ax=axes[0],
                 color='red',
                 label='ADMM-mTRF',
                 linewidth=3,
                 marker='*',
                 markersize=8,
                 markeredgecolor='black', markeredgewidth=1.5)
    axes[0].set_yscale('log')
    axes[0].set_title('CPU time', fontsize=12)
    axes[0].set_xlabel('number of electrodes (N)', fontsize=12)
    axes[0].set_ylabel('run-time (sec)', fontsize=12)
    axes[0].legend(fontsize=10)

    # Plot memory results
    sns.lineplot(x=np.linspace(1,total_ch, total_ch),
                 y=naplib_mems,
                 ax=axes[1],
                 color='purple',
                 label='NapLib',
                 linewidth=3,
                 marker='o',
                 markersize=8,
                 markeredgecolor='black', markeredgewidth=1.5)
    sns.lineplot(x=np.linspace(1,total_ch, total_ch),
                 y=admm_mems,
                 ax=axes[1],
                 color='red',
                 label='ADMM-mTRF',
                 linewidth=3,
                 marker='*',
                 markersize=8,
                 markeredgecolor='black', markeredgewidth=1.5)
    axes[1].set_title('Memory Usage', fontsize=12)
    axes[1].set_xlabel('number of electrodes (N)', fontsize=12)
    axes[1].set_ylabel('Max maemory used (MiB)', fontsize=12)
    axes[1].legend(fontsize=10)

    plt.show()




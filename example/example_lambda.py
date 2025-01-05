import matplotlib.pyplot as plt
from scipy.signal import resample
from sklearn.linear_model import Ridge, ElasticNet
import numpy as np

import naplib as nl
from naplib.visualization import strf_plot

from ADMM_mTRF.model import admm_mTRF

import time
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
    data['resp'] = [d[:,9:10] for d in data['resp']]

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

    # leave out 1 trial for testing
    data_train = data[:-1]
    data_test = data[-1:]

    l1_ratios = [1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 2e-1, 5e-1]
    fig, axes = plt.subplots(len(l1_ratios),2,figsize=(15,2.5))

    for ll, l1_ratio in enumerate(l1_ratios):
        # Fit STRF models using ElasticNet (L1 and L2 penalty) regression.
        # define the estimator to be used in this TRF model
        t0 = time.time()
        estimator = ElasticNet(l1_ratio=l1_ratio)

        strf_naplib = nl.encoding.TRF(tmin, tmax, sfreq, estimator=estimator)
        strf_naplib.fit(data=data_train, X='spec_32', y='resp')
        t1 = time.time()
        naplib_time = t1 - t0

        # -----------------------------------------------------
        # Fit STRF Models with ElasticNet -- ADMM implementaion
        # -----------------------------------------------------
        # define the ADMM-mTRF model
        t0 = time.time()
        admm_mtrf = admm_mTRF(direction=1)
        admm_mtrf.train(data_train['spec_32'], data_train['resp'],
                           sfreq, tmin, tmax,
                           alpha=1.0,
                           l1_ratio=l1_ratio)
        t1 = time.time()
        admm_time = t1 - t0


        # -------------------------
        # Analyze the STRFs Weights
        # -------------------------
        # we can access the STRF weights through the .coef_ attribute of the model
        coef_naplib    = strf_naplib.coef_
        # make the shape same as naplib
        coef_admm_mtrf = admm_mtrf.weights.transpose((2,0,1)).copy()/sfreq

        print(f'STRF shape (num_outputs, frequency, lag) = {coef_naplib.shape}')
        print(f'ADMM_mTRF shape (num_outputs, frequency, lag) = {coef_admm_mtrf.shape}')
        print('*'*30)
        print(f"naplib total time: {naplib_time:1.3f}")
        print(f"admm total time: {admm_time:1.3f}")

        # Now, visualize the STRF weights for the last electrode and for each model
        freqs = [171, 5000]
        elec = 0
        max_err = np.max(np.abs(coef_admm_mtrf[elec]-coef_naplib[elec]))
        print(f"maximum abs error: {max_err:1.2e}")

        strf_plot(coef_admm_mtrf[elec], tmin=tmin, tmax=tmax, freqs=freqs, ax=axes[ll,0])
        axes[ll,0].set_title(f'ADMM-mTRF, {l1_ratio:1.2e}')

        # Compute and plot STRF predictions.
        predictions = admm_mtrf.predict(data_test['spec_32'])
        # plot the predictions for the first 10 seconds of the final trial for the last electrode
        axes[ll,1].plot(data_test['resp'][-1][:1000,elec], label='neural')
        axes[ll,1].plot(predictions[-1][:1000,elec],'r',label=f'ADMM-mTRF pred')
        plt.xticks([0, 500, 1000], ['0', '5', '10'])
        plt.title('ADMM-mTRF prediction')

    fig.tight_layout()
    plt.show()






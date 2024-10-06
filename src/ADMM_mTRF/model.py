import numpy as np

from scipy.linalg import cholesky, solve_triangular

from pywt import threshold


# trf dependencies
from mtrf.model import TRF
from mtrf.stats import (
    _crossval,
    _progressbar,
    _check_k,
    neg_mse,
    pearsonr,
)
from mtrf.matrices import (
    lag_matrix,
    _check_data,
    _get_xy,
)


def covariance_matrices(x, y, lags, zeropad=True, preload=True):
    """
    Compute (auto-)covariance of x and y.

    Compute the autocovariance of the time-lagged input x and the covariance of
    x and the output y. When passed a list of trials for x, and y, covariance
    matrices will be computed for each trial.

    Parameters
    ----------
    x: numpy.ndarray or list
        Input data in samples-by-features array or list of such arrays.
    y: numpy.ndarray or list
        Output data in samples-by-features array or list of such arrays.
    lags: list or numpy.ndarray
        Time lags in samples.
    zeropad: bool
        If True (default), pad the input with zeros, if false, truncate the output.

    Returns
    -------
    cov_xx: numpy.ndarray
        Three dimensional autocovariance matrix. 1st dimension's size is the number
        of trials, 2nd and 3rd dimensions' size is lags times features in x.
        If x contains only one trial, the first dimension is empty and will be removed.
    cov_xy: numpy.ndarray
        Three dimensional x-y-covariance matrix. 1st dimension's size is the number
        of trials, 2nd dimension's size is lags times features in x and 3rd dimension's
        size is features in y. If y contains only one trial, the first dimension is
        empty and will be removed.
    """
    x, y, _ = _check_data(x, y)
    if zeropad is False:
        y = truncate(y, lags[0], lags[-1])
    cov_xx, cov_xy = 0, 0
    for i, (x_i, y_i) in enumerate(zip(x, y)):
        x_lag = lag_matrix(x_i, lags, zeropad)
        if preload is True:
            if i == 0:
                cov_xx = np.zeros((len(x), x_lag.shape[-1], x_lag.shape[-1]))
                cov_xy = np.zeros((len(y), x_lag.shape[-1], y_i.shape[-1]))
            cov_xx[i] = (1/x_lag.shape[0]) * x_lag.T @ x_lag
            cov_xy[i] = (1/x_lag.shape[0]) * x_lag.T @ y_i
        else:
            cov_xx += (1/x_lag.shape[0]) * x_lag.T @ x_lag
            cov_xy += (1/x_lag.shape[0]) * x_lag.T @ y_i
    if preload is False:
        cov_xx, cov_xy = cov_xx / len(x), cov_xy / len(x)

    return cov_xx, cov_xy


class admm_mTRF(TRF):
    def __init__(self,
                 direction = 1,
                 kind = "multi",
                 zeropad = True,
                 method = "elasticnet",
                 preload = True,
                 metric = pearsonr):
        self.weights = None
        self.bias = None
        self.times = None
        self.fs = None
        self.regularization = None
        if not callable(metric):
            raise ValueError("Metric function must be callable")
        else:
            self.metric = metric
        if isinstance(preload, bool):
            self.preload = preload
        else:
            raise ValueError("Parameter preload must be either True or False!")
        if direction in [1, -1]:
            self.direction = direction
        else:
            raise ValueError("Parameter direction must be either 1 or -1!")
        if kind in ["multi", "single"]:
            self.kind = kind
        else:
            raise ValueError('Paramter kind must be either "multi" or "single"!')
        if isinstance(zeropad, bool):
            self.zeropad = zeropad
        else:
            raise ValueError("Parameter zeropad must be boolean!")
        if method in ["ridge", "tikhonov", "banded"]:
            raise ValueError('Please use the original mTRFpy toolbox')
        elif method in ["elasticnet"]:
            self.method = method
        else:
            raise ValueError('Method must be "elasticnet"')

    def train(
        self,
        stimulus,
        response,
        fs,
        tmin,
        tmax,
        alpha = 1.0,
        l1_ratio = 0.01,
        opt_param = dict(),
        bands=None,
        k=-1,
        average=True,
        seed=None,
        verbose=True,
    ):
        """
        Optimize the regularization parameter.

        For each value in `regularization`, a model is trained and validated using
        k-fold cross validation. Then the best regularization value (i.e. the one
        that results in the highest mean metric value is chosen to train the final model.

        Compute a linear mapping between stimulus and response, or vice versa, using
        regularized regression. This method can compare multiple values for the
        regularization parameter and select the best one (i.e. the one that yields
        the highest prediction metric).

        Parameters
        ----------
        stimulus: list
            Each element must contain one trial's stimulus in a two-dimensional
            samples-by-features array (second dimension can be omitted if there is
            only a single feature.
        response: list
            Each element must contain one trial's response in a two-dimensional
            samples-by-channels array.
        fs: int
            Sample rate of stimulus and response in hertz.
        tmin: float
            Minimum time lag in seconds.
        tmax: float
            Maximum time lag in seconds.
        regularization: list or float or int
            Values for the regularization parameter lambda. The model is fitted
            separately for each value and the one yielding the highest accuracy
            is chosen (correlation and mean squared error of each model are returned).
        opt_param: dictionary of admm parameters
        bands: list or None
            Must only be provided when using banded ridge regression. Size of the
            features for which a regularization parameter is fitted, in the order they
            appear in the stimulus matrix. For example, when the stimulus consists of
            an envelope vector and a 16-band spectrogram, bands would be [1, 16].
        k: int
            Number of data splits for cross validation, defaults to -1 (leave-one-out).
        average: bool or list or numpy.ndarray
            If True (default), average metric cross all predictions (e.g. channels in the
            case of forward modelling). If `average` is an array of indices only average
            those features.
        seed: int
            Seed for the random number generator.
        verbose: bool
            If True (default), show a progress bar during fitting.

        Returns
        -------
        metric : list
            When providing multiple `regularization` values this returns the metric as
            computed by the metric function defined in the `TRF.metric` attribute
            for every regularization value.
        """
        if average is False:
            raise ValueError("Average must be True or a list of indices!")
        stimulus, response, n_trials = _check_data(stimulus, response)
        if not np.isscalar(l1_ratio):
            k = _check_k(k, n_trials)
        x, y, tmin, tmax = _get_xy(stimulus, response, tmin, tmax, self.direction)
        lags = list(range(int(np.floor(tmin * fs)), int(np.ceil(tmax * fs)) + 1))
        if np.isscalar(l1_ratio):
            self._train(x, y, fs, tmin, tmax,
                        alpha=alpha, l1_ratio=l1_ratio, opt_param=opt_param)
            return
        else:  # run cross-validation once per regularization parameter
            raise NotImplementedError
            # pre-compute covariance matrices
            cov_xx, cov_xy = None, None
            if self.preload:
                cov_xx, cov_xy = covariance_matrices(
                    x, y, lags, self.zeropad, self.preload
                )
            else:
                cov_xx, cov_xy = None, None
            metric = np.zeros(len(regularization))
            for ir in _progressbar(
                range(len(regularization)),
                "Hyperparameter optimization",
                verbose=verbose,
            ):
                metric[ir] = _crossval(
                    self.copy(),
                    x,
                    y,
                    cov_xx,
                    cov_xy,
                    lags,
                    fs,
                    regularization[ir],
                    k,
                    seed=seed,
                    average=average,
                    verbose=verbose,
                )
            best_regularization = list(regularization)[np.argmax(metric)]
            self._train(x, y, fs, tmin, tmax, best_regularization)
            return metric


    def _train(self, x, y, fs, tmin, tmax, alpha=1.0, l1_ratio=0.01, opt_param=dict()):
        '''
        _train code from mTRFpy, adapted to use the cholesky factorization
        instead of matrix inversion
        '''
        #set optimization param
        opt_defaults = {'rho':0.01,
                        'max_iter':5000,
                        'verbose':True,
                        'tol':1e-5}
        for key, value in opt_defaults.items():
            if key not in opt_param:
                opt_param[key] = value
        rho, max_iter, verbose, tol = opt_param.values()
        self.fs = fs
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.regularization = None
        self.tmin = tmin
        self.tmax = tmax
        # get cov
        lags = list(range(int(np.floor(tmin * fs)), int(np.ceil(tmax * fs)) + 1))
        cov_xx, cov_xy = covariance_matrices(x, y, lags, self.zeropad, preload=False)
        # setup regularization mat
        regmat = np.identity(cov_xx.shape[1])
        regmat[0,0] = 0.0
        regmat *= (alpha * (1-l1_ratio) + rho)
        thresh_val = alpha*l1_ratio/rho
        # init variables
        # use chol for l2-solve
        L = cholesky(cov_xx + regmat, lower=True)
        Q = solve_triangular(L, cov_xy, lower=True)
        w1 = solve_triangular(L.T, Q, lower=False)
        w2 = threshold(w1, thresh_val, mode='soft')
        u  = w1-w2
        flag_tol_reached = False
        for itr in range(max_iter):
            Q = solve_triangular(L, cov_xy + rho*(w2-u), lower=True)
            w1 = solve_triangular(L.T, Q, lower=False)
            w2 = threshold(w1+u, thresh_val, mode='soft')
            u  = u+w1-w2
            err = np.linalg.norm(w1-w2)/w1.shape[1]
            if verbose:
                if itr%500==0:
                    print(f'itr:{itr:3d}, err:{err:1.3e}')
            if err<=tol:
                flag_tol_reached = True
                break
        if verbose:
            print(f'done in {itr}, tol reached: {flag_tol_reached}')
        if flag_tol_reached:
            weight_matrix = w2 /(1/self.fs)
        else:
            weight_matrix = w1 /(1/self.fs)
        self.bias = weight_matrix[0:1]
        if self.bias.ndim == 1:  # add empty dimension for single feature models
            self.bias = np.expand_dims(self.bias, axis=0)
        self.weights = weight_matrix[1:].reshape(
            (x[0].shape[1], len(lags), y[0].shape[1]), order="F"
        )
        self.times = np.array(lags) / fs
        self.fs = fs




import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from statsmodels.nonparametric.kde import KDEUnivariate

from SALib.sample import sobol_sequence


def acq_min(ac, gp, y_min, bounds, random_state, n_warmup=None,
            n_iter=20, warmup_points: np.array = None, n_returns=1,
            x_min=None):
    """
    A function to find the minimum of the acquisition function

    It uses a combination of random sampling (cheap) and the 'L-BFGS-B'
    optimization method. First by sampling `n_warmup` (1e5) points at random,
    and then running L-BFGS-B from `n_iter` (250) random starting points.

    Parameters
    ----------
    :param n_returns:
        Number of best hits to return

    :param ac:
        The acquisition function object that return its point-wise value.

    :param gp:
        A gaussian process or a wrapper around a list of gaussian processes fitted to the relevant data.

    :param x_min:
        The position of the current minimum known value of the target function.
        Formatted as a 1D array

    :param y_min:
        The current minimum known value of the target function.

    :param bounds:
        The variables bounds to limit the search of the acq min.

    :param random_state:
        instance of np.RandomState random number generator

    :param n_warmup:
        number of times to randomly sample the aquisition function

    :param n_iter:
        number of times to run scipy.minimize

    :param warmup_points:
        pre-supplied points for the warmup

    Returns
    -------
    :return: x_min, The arg min of the acquisition function.
    """

    if n_warmup is None:
        n_warmup = 20000

    # Warm up with random points
    if warmup_points is None:
        x = create_sobol(n_warmup, np.random.randint(n_warmup * 3), bounds)
    else:
        x = warmup_points

    x += np.random.uniform(-0.001, 0.001, size=x.shape)

    def warmup(x_tries, gp, y_min):
        return ac(x_tries, gp=gp, y_min=y_min).flatten()

    y = warmup(x, gp, y_min)
    x_sorted = x[y.argsort()]

    x_seeds = x_sorted[:n_iter]

    x_searched = []
    y_searched = []

    for i, x_try in enumerate(x_seeds):
        # Find the minimum of minus the acquisition function
        res = minimize(lambda x: ac(x.reshape(1, -1), gp=gp, y_min=y_min).flatten(),
                       x_try,
                       bounds=bounds,
                       method="L-BFGS-B")

        # See if success
        if not res.success:
            continue

        x_searched.append(res.x)
        y_searched.append(res.fun)

    if len(x_searched) > 0:
        x_searched = np.array(x_searched)
        y_searched = np.array(y_searched).reshape(-1)

        try:
            x = np.append(x, x_searched, axis=0)
        except ValueError as e:
            raise ValueError(f"all the input arrays must have same number of dimensions"
                             f"x:{x}\n"
                             f"x_searched:{x_searched}") from e
        y = np.append(y, y_searched, axis=0)

    x_final_sorted = x[y.argsort()]

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    return np.clip(x_final_sorted[:n_returns], bounds[:, 0], bounds[:, 1])


class UtilityFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, kind, kappa, xi, n_samples=100):
        """
        If UCB is to be used, a constant kappa is needed.
        """
        self.kappa = kappa

        self.xi = xi

        self.n_samples = n_samples

        if kind not in ['ucb', 'ei', 'poi', "ei_manual", 'ucb_manual']:
            err = "The utility function " \
                  f"{kind} has not been implemented, " \
                  "please choose one of ucb, ei, or poi."
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def utility(self, x, gp, y_min, n_samples=0):
        if n_samples == 0:
            n_samples = self.n_samples
        if self.kind == 'ucb':
            return self._ucb(x, gp, self.kappa)
        if self.kind == 'ucb_manual':
            return self._ucb_manual(x, gp, self.kappa, n_samples)
        if self.kind == 'ei':
            return self._ei(x, gp, y_min, self.xi)
        if self.kind == 'ei_manual':
            return self._ei_manual(x, gp, y_min, self.xi, n_samples)
        if self.kind == 'poi':
            return self._poi(x, gp, y_min, self.xi)

    @staticmethod
    def _ucb(x, gp, kappa):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        return abs(mean) - kappa * std

    @staticmethod
    def _ucb_folded(x, gp, kappa):
        """
        upper confidence bound for a normal distribution that is folded at zero
        :param x:
        :param gp:
        :param kappa:
        :return:
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        return abs(mean) - kappa * std

    @staticmethod
    def _ucb_manual(x, gp, kappa, n_samples=100):
        """
        Upper confidence bound for non-gaussian distributions.
        :param x: x coordinates to be evaluated
        :param gp: gaussian process handle
        :param kappa: exploration-exploitation tradeoff. In this implementation it works by setting an upper bound for
        the cumulative distribution function equivalent to the upper bound derived from a normal cdf at kappa.
        The last entry still within the cdf upper limit is chosen as the minimum value within the upper confidence bound
        :param n_samples: number of samples to be drawn from the underlying gp-distribution over which the density
        function is approximated
        :return: upper confidence utility at x
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            gp_dist_samples = gp.sample_y(x, n_samples=n_samples)
            kernel_list = [KDEUnivariate(gp_dist_samples[i, :]) for i in range(gp_dist_samples.shape[0])]
            for k in kernel_list:
                k.fit()

            densities = np.array([kde.density / np.sum(kde.density) for kde in kernel_list])
            supports = np.array([kde.support for kde in kernel_list])

            cdf = np.cumsum(densities, axis=1)
            cdf[cdf > norm.cdf(kappa)] = 0

            argmin_indixes = cdf.argmin(axis=1)
            ret = supports[np.arange(supports.shape[0]), argmin_indixes]
        return ret

    @staticmethod
    def _ei(x, gp, y_min, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        z = (mean - y_min - xi) / std
        return (mean - y_min - xi) * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def _ei_manual(x, gp, y_min, xi, n_samples):
        """
        Expected improvement for non-gaussian distributions which can occour when multiple gp's are aggregated and/or
        transformed.
        The distribution is sampled n_samples times and then approximated using the KDEUnivariate from
        statsmodels.nonparametric.kde. This kde was used because it fit's faster than the kde from scipy because it uses
        FFT and because it directly computes support and density points which can then be used to approximate the
        integration and save time.

        Sum to aproximate the integration according to page 2 of
        https://www.cse.wustl.edu/~garnett/cse515t/spring_2015/files/lecture_notes/12.pdf
        with added xi according to http://krasserm.github.io/2018/03/21/bayesian-optimization/

        As a test the y_pre[y_pre < 0] = 0 was replaced with y_pre[y_pre < 0] = y_pre[y_pre < 0] / 100 to give the
        acq_min L-BFGS-B search algorithm more to work with

        :param x: x coordinates to be evaluated
        :param gp: gaussian process handle
        :param y_min: highest y-value observed so far
        :param xi: exploration - exploitation tradeoff
        :param n_samples: number of samples to be drawn from the underlying gp-distribution over which the density
        function is approximated
        :return: expected improvement at x
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp_dist_samples = gp.sample_y(x, n_samples=n_samples, random_state=np.random.randint(0, 265))
            # gp_dist_samples = gp.sample_y(x, n_samples=n_samples)

            kernel_list = [KDEUnivariate(gp_dist_samples[i, :]) for i in range(gp_dist_samples.shape[0])]
            for k in kernel_list:
                k.fit()

            y_pre = np.array(
                [np.multiply(kde.support - y_min - xi, kde.density / np.sum(kde.density)) for kde in kernel_list])
            y_pre[y_pre < 0] = y_pre[y_pre < 0] / 10000

            ei = y_pre.sum(axis=1)
        return ei

    @staticmethod
    def _poi(x, gp, y_min, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        z = (mean - y_min - xi) / std
        return norm.cdf(z)


def load_logs(optimizer, logs):
    """Load previous ...

    """
    import json

    if isinstance(logs, str):
        logs = [logs]

    for log in logs:
        with open(log, "r") as j:
            while True:
                try:
                    iteration = next(j)
                except StopIteration:
                    break

                iteration = json.loads(iteration)
                try:
                    optimizer.register(
                        params=iteration["params"],
                        target=iteration["target"],
                    )
                except KeyError:
                    pass

    return optimizer


class Colours:
    """Print in nice colours."""

    BLUE = '\033[94m'
    BOLD = '\033[1m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    END = '\033[0m'
    GREEN = '\033[92m'
    PURPLE = '\033[95m'
    RED = '\033[91m'
    UNDERLINE = '\033[4m'
    YELLOW = '\033[93m'

    @classmethod
    def _wrap_colour(cls, s, colour):
        return colour + s + cls.END

    @classmethod
    def black(cls, s):
        """Wrap text in blue."""
        return cls._wrap_colour(s, cls.END)

    @classmethod
    def blue(cls, s):
        """Wrap text in blue."""
        return cls._wrap_colour(s, cls.BLUE)

    @classmethod
    def bold(cls, s):
        """Wrap text in bold."""
        return cls._wrap_colour(s, cls.BOLD)

    @classmethod
    def cyan(cls, s):
        """Wrap text in cyan."""
        return cls._wrap_colour(s, cls.CYAN)

    @classmethod
    def darkcyan(cls, s):
        """Wrap text in darkcyan."""
        return cls._wrap_colour(s, cls.DARKCYAN)

    @classmethod
    def green(cls, s):
        """Wrap text in green."""
        return cls._wrap_colour(s, cls.GREEN)

    @classmethod
    def purple(cls, s):
        """Wrap text in purple."""
        return cls._wrap_colour(s, cls.PURPLE)

    @classmethod
    def red(cls, s):
        """Wrap text in red."""
        return cls._wrap_colour(s, cls.RED)

    @classmethod
    def underline(cls, s):
        """Wrap text in underline."""
        return cls._wrap_colour(s, cls.UNDERLINE)

    @classmethod
    def yellow(cls, s):
        """Wrap text in yellow."""
        return cls._wrap_colour(s, cls.YELLOW)


def create_sobol(n_warmup_points, n_random, bounds):
    warmup_points = sobol_sequence.sample(n_warmup_points + n_random, bounds.shape[0])

    # shuffle dimensions
    warmup_points = warmup_points.T
    np.random.shuffle(warmup_points)
    warmup_points = warmup_points.T

    warmup_points = warmup_points[n_random:, :] * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    return warmup_points

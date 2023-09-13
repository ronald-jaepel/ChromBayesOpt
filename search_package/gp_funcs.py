import traceback
import numpy as np
from datetime import datetime
from joblib import Parallel, delayed
from logging import getLogger
import pandas as pd
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.utils import check_random_state
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter, _approx_fprime

from search_package.cadet_interface import plot_sim
from search_package.helper_functions import save_time_df
from search_package.bayes_opt_custom import UtilityFunction


class GaussianListWrapper:
    def __init__(self, gp_list, aggregation_function, variance_aggregation_function,
                 covariance_aggregation_function=None, transform=False, mask=None, bounds=None,
                 use_internal_parallelization=False, backend="loky", filter_x_to_unit_cube=False,
                 use_uncertainty_as_weights=False):
        """

        :param gp_list:
        :param aggregation_function:
        :param variance_aggregation_function:
        :param covariance_aggregation_function:
        :param transform:
        :param mask: boolean np.array of shape (n_features, len(gp_list)).
        True means the feature has an effect on
        the respective gp
        :param bounds:
        """
        self.gp_list = gp_list
        self.target_dims = len(gp_list)
        self.mean_agg_func = aggregation_function
        self.var_agg_func = variance_aggregation_function
        self.cov_agg_func = covariance_aggregation_function
        self._transform = transform
        self.input_transformation_x = lambda x: x
        self.input_transformation_x_inv = lambda x: x
        self.input_transformation_y = lambda x: x
        self.y_std = [1]
        self.mask = mask
        self.filter_x_to_unit_cube = filter_x_to_unit_cube
        self.use_uncertainty_as_weights = use_uncertainty_as_weights
        self.parallel = use_internal_parallelization
        self.filtered_x_shape = (None, None)
        if use_internal_parallelization:
            self.par_pool = Parallel(n_jobs=len(self.gp_list), backend=backend, verbose=0)
        else:
            self.par_pool = None
        if bounds is not None and transform is True:
            self._add_input_transformation_x(bounds)
        elif filter_x_to_unit_cube:
            try:
                raise ValueError("filter_x_to_unit_cube was set without setting "
                                 "transform and boundarys")
            except ValueError:
                traceback.print_exc()

    def _add_input_transformation_x(self, bounds):
        lower, upper = zip(*[(up, low) for key, (up, low) in sorted(bounds.items())])
        m = np.array(upper) - np.array(lower)
        m[m == 0] = 1
        b = np.array(lower)
        self.input_transformation_x_inv = lambda x: x * m + b
        self.input_transformation_x = lambda x: (x - b) / m

    def input_transformation_y_inv(self, y_data, index=None):
        # if index is not None:
        #     try:
        #         y_data = y_data * self.y_std[index]
        #     except TypeError as e:
        #         raise TypeError("y_std in the Gaussian Process is still an integer. Are you sure"
        #                         "that you've fitted your gp before making a prediction?") from e
        # else:
        #     y_data = y_data * self.y_std
        return y_data

    def _add_input_transformation_y(self, y):
        self.input_transformation_y = lambda x: x

    def fit(self, x, y):
        """Fit all Gaussian process regression models in the list.

        Parameters
        ----------
        x : array-like, shape = (n_samples, n_features)
            Training data

        y : array-like, shape = (n_samples, [n_output_dims])
            Target values

        Returns
        -------
        self : returns an instance of self.
        """
        x = self.input_transformation_x(x)
        if self.filter_x_to_unit_cube and self._transform:
            filter = np.all(np.abs(x - 0.5) <= 0.5, 1)
            y = y[filter]
            x = x[filter]
            self.filtered_x_shape = x.shape
        self._add_input_transformation_y(y)
        y = self.input_transformation_y(y)

        x_masked = []
        for i in range(len(self.gp_list)):
            x_masked.append(x[:, self.mask[:, i]])

        if self.parallel:
            fitted_gp_list = self.par_pool(
                delayed(gp.fit)(x_masked[i], y[:, i]) for i, gp in enumerate(self.gp_list))
        else:
            i = 0
            fitted_gp_list = [gp.fit(x_masked[i], y[:, i]) for i, gp in enumerate(self.gp_list)]
        self.gp_list = fitted_gp_list
        # for gp in self.gp_list:
        #     # print(gp)
        #     print(gp.kernel_)
        return self

    def predict(self, x, return_std=False, return_cov=False, return_full_target_space=False):
        """Predict using the Gaussian process regression model

                We can also predict based on an unfitted model by using the GP prior.
                In addition to the mean of the predictive distribution, also its
                standard deviation (return_std=True) or covariance (return_cov=True).
                Note that at most one of the two can be requested.

                Parameters
                ----------
                x : array-like, shape = (n_samples, n_features)
                    Query points where the GP is evaluated

                return_std : bool, default: False
                    If True, the standard-deviation of the predictive distribution at
                    the query points is returned along with the mean.

                return_cov : bool, default: False
                    If True, the covariance of the joint predictive distribution at
                    the query points is returned along with the mean

                return_full_target_space : bool, default: False
                    If True, the entire target space is returned, i.e. all results of all
                    individual gaussian process regressors in the list

                Returns
                -------
                y_mean : array, shape = (n_samples, [n_output_dims])
                    Mean of predictive distribution a query points

                y_std : array, shape = (n_samples,), optional
                    Standard deviation of predictive distribution at query points.
                    Only returned when return_std is True.

                y_cov : array, shape = (n_samples, n_samples), optional
                    Covariance of joint predictive distribution a query points.
                    Only returned when return_cov is True.
                """
        x = self.input_transformation_x(x)

        x_masked = []
        for i in range(len(self.gp_list)):
            x_masked.append(x[:, self.mask[:, i]])

        if self.parallel and x.shape[0] > 1:
            results = self.par_pool(delayed(gp.predict)(x_masked[i], return_std, return_cov)
                                    for i, gp in enumerate(self.gp_list))
        else:
            results = [gp.predict(x_masked[i], return_std, return_cov) for i, gp in
                       enumerate(self.gp_list)]

        if not return_full_target_space:
            if return_cov:
                raise RuntimeError("Covariance not implemented")
            elif return_std:
                y_means, y_vars = zip(*results)
                y_means = np.array(y_means).T
                y_means = self.input_transformation_y_inv(y_means)
                y_vars = np.array(y_vars).T
                y_vars = self.input_transformation_y_inv(y_vars)
                if self.use_uncertainty_as_weights:
                    y_mean = self.mean_agg_func(y_means, y_vars)
                else:
                    y_mean = self.mean_agg_func(y_means)
                y_var = self.var_agg_func(y_means, y_vars)
                return y_mean, y_var
            else:
                y_means = results
                y_means = np.array(y_means).T
                y_means = self.input_transformation_y_inv(y_means)
                if self.use_uncertainty_as_weights:
                    raise RuntimeError("Why is this called?")
                y_mean = self.mean_agg_func(y_means)
                return y_mean
        else:
            y_means = results
            y_means = np.array(y_means).T
            y_means = self.input_transformation_y_inv(y_means)
            y_mean = self.mean_agg_func(y_means)
            if self.use_uncertainty_as_weights:
                raise RuntimeError("Why is this called?")
            return np.append(np.array(y_means), np.array([y_mean]))

    def predict_individual_gp(self, x, gp_index, return_std=False, return_cov=False):
        """Predict using the Gaussian process regression model

                We can also predict based on an unfitted model by using the GP prior.
                In addition to the mean of the predictive distribution, also its
                standard deviation (return_std=True) or covariance (return_cov=True).
                Note that at most one of the two can be requested.

                Parameters
                ----------
                x : array-like, shape = (n_samples, n_features)
                    Query points where the GP is evaluated

                return_std : bool, default: False
                    If True, the standard-deviation of the predictive distribution at
                    the query points is returned along with the mean.

                return_cov : bool, default: False
                    If True, the covariance of the joint predictive distribution at
                    the query points is returned along with the mean

                Returns
                -------
                y_mean : array, shape = (n_samples, [n_output_dims])
                    Mean of predictive distribution a query points

                y_std : array, shape = (n_samples,), optional
                    Standard deviation of predictive distribution at query points.
                    Only returned when return_std is True.

                y_cov : array, shape = (n_samples, n_samples), optional
                    Covariance of joint predictive distribution a query points.
                    Only returned when return_cov is True.
                """
        x = self.input_transformation_x(x)

        x_masked = []
        for i in range(len(self.gp_list)):
            x_masked.append(x[:, self.mask[:, i]])

        if return_cov:
            y_mean, y_cov = self.gp_list[gp_index].predict(x_masked[gp_index], return_std,
                                                           return_cov)
            y_mean = self.input_transformation_y_inv(y_mean, index=gp_index)
            return y_mean, y_cov
        elif return_std:
            y_mean, y_var = self.gp_list[gp_index].predict(x_masked[gp_index], return_std,
                                                           return_cov)
            y_mean = self.input_transformation_y_inv(y_mean, index=gp_index)
            y_var = self.input_transformation_y_inv(y_var, index=gp_index)
            return y_mean, y_var
        else:
            y_mean = self.gp_list[gp_index].predict(x_masked[gp_index], return_std, return_cov)
            y_mean = self.input_transformation_y_inv(y_mean, index=gp_index)
            return y_mean

    def sample_y(self, x, n_samples=1, random_state=0):
        """Draw samples from Gaussian process and evaluate at X.

                Parameters
                ----------
                x : array-like, shape = (n_samples_X, n_features)
                    Query points where the GP samples are evaluated

                n_samples : int, default: 1
                    The number of samples drawn from the Gaussian process

                random_state : int, RandomState instance or None, optional (default=0)
                    If int, random_state is the seed used by the random number
                    generator; If RandomState instance, random_state is the
                    random number generator; If None, the random number
                    generator is the RandomState instance used by `np.random`.

                Returns
                -------
                y_samples : array, shape = (n_samples_X, [n_output_dims], n_samples)
                    Values of n_samples samples drawn from Gaussian process and
                    evaluated at query points.
                """
        x = self.input_transformation_x(x)

        x_masked = []
        for i in range(len(self.gp_list)):
            x_masked.append(x[:, self.mask[:, i]])

        rng = check_random_state(random_state)
        samples_list = []
        for i, gp in enumerate(self.gp_list):
            y_samples = gp.sample_y(x_masked[i], n_samples, random_state=rng)
            y_samples = self.input_transformation_y_inv(y_samples)
            samples_list.append(y_samples)
        samples_array = np.array(samples_list).transpose((1, 0, 2))
        y_samples = self.mean_agg_func(samples_array).reshape(x.shape[0], n_samples)
        return np.array(y_samples)

    def log_marginal_likelihood(self, theta=None, eval_gradient=False, return_array=True):
        """Returns log-marginal likelihood of theta for training data.

        Parameters
        ----------
        theta : array-like, shape = (n_kernel_params,) or None
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated. If None, the precomputed log_marginal_likelihood
            of ``self.kernel_.theta`` is returned.

        eval_gradient : bool, default: False
            If True, the gradient of the log-marginal likelihood with respect
            to the kernel hyperparameters at position theta is returned
            additionally. If True, theta must not be None.

        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of theta for training data.

        log_likelihood_gradient : array, shape = (n_kernel_params,), optional
            Gradient of the log-marginal likelihood with respect to the kernel
            hyperparameters at position theta.
            Only returned when eval_gradient is True.
        """
        if eval_gradient:
            log_likelihoods, log_likelihood_gradients = zip(
                *[gp.log_marginal_likelihood(theta, eval_gradient) for gp in self.gp_list])
            log_likelihood = np.mean(log_likelihoods)
            log_likelihood_gradient = np.mean(log_likelihood_gradients)
            return log_likelihood, log_likelihood_gradient
        else:
            log_likelihoods = [gp.log_marginal_likelihood(theta, eval_gradient) for gp in
                               self.gp_list]
            if return_array:
                return np.array(log_likelihoods)
            else:
                log_likelihood = np.mean(log_likelihoods)
                return log_likelihood

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        theta_opts, func_mins = zip(
            *[gp._constrained_optimization(obj_func, initial_theta, bounds) for gp in self.gp_list])
        theta_opt = np.mean(theta_opts)
        func_min = np.mean(func_mins)
        return theta_opt, func_min

    def set_params(self, **params):
        self.gp_list = [gp.set_params(**params) for gp in self.gp_list]
        return self


class ArcCosine(Kernel):
    """
    Mimiced from GPflow
    The Arc-cosine family of kernels which mimics the computation in neural
    networks. The order parameter specifies the assumed activation function.
    The Multi Layer Perceptron (MLP) kernel is closely related to the ArcCosine
    kernel of order 0. The key reference is

    ::

        @incollection{NIPS2009_3628,
            title = {Kernel Methods for Deep Learning},
            author = {Youngmin Cho and Lawrence K. Saul},
            booktitle = {Advances in Neural Information Processing Systems 22},
            year = {2009},
            url = {http://papers.nips.cc/paper/3628-kernel-methods-for-deep-learning.pdf}
        }
    """

    def __init__(self, order=0, variance=1.0, weight_variances=1., bias_variance=1.,
                 variance_bounds=(1e-5, 1e5), weight_variances_bounds=(1.e-5, 1e5),
                 bias_variance_bounds=(1.e-5, 1e5)):
        """
        - order specifies the activation function of the neural network
          the function is a rectified monomial of the chosen order.
        - variance is the initial value for the variance parameter
        - weight_variances is the initial value for the weight_variances parameter
          defaults to 1.0
        - bias_variance is the initial value for the bias_variance parameter
          defaults to 1.0.
        """
        self.implemented_orders = {0, 1, 2}

        if order not in self.implemented_orders:
            raise NotImplementedError('Requested kernel order is not implemented. \n '
                                      'Please choose from [0, 1, 2]')
        self.order = order
        self.variance = variance
        self.weight_variances = weight_variances
        self.bias_variance = bias_variance
        self.variance_bounds = variance_bounds
        self.weight_variances_bounds = weight_variances_bounds
        self.bias_variance_bounds = bias_variance_bounds

    @property
    def hyperparameter_variance(self):
        return Hyperparameter("variance", "numeric", self.variance_bounds)

    @property
    def hyperparameter_weight_variances(self):
        return Hyperparameter("weight_variances", "numeric", self.weight_variances_bounds)

    @property
    def hyperparameter_bias_variance(self):
        return Hyperparameter("bias_variance", "numeric", self.bias_variance_bounds)

    def _weighted_product(self, X, Y=None):
        if Y is None:
            return np.sum(self.weight_variances * np.square(X), axis=-1) + self.bias_variance
        return np.dot((self.weight_variances * X), Y.T) + self.bias_variance

    def _J(self, theta):
        """
        Implements the order dependent family of functions defined in equations
        4 to 7 in the reference paper.
        """
        if self.order == 0:
            return np.pi - theta
        elif self.order == 1:
            return np.sin(theta) + (np.pi - theta) * np.cos(theta)
        elif self.order == 2:
            return 3. * np.sin(theta) * np.cos(theta) + \
                   (np.pi - theta) * (1. + 2. * np.cos(theta) ** 2)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """

        X_denominator = np.sqrt(self._weighted_product(X))
        if Y is None:
            Y = X
            Y_denominator = X_denominator
        else:
            Y_denominator = np.sqrt(self._weighted_product(Y))

        numerator = self._weighted_product(X, Y)
        X_denominator = np.expand_dims(X_denominator, -1)
        Y_denominator = np.matrix.transpose(np.expand_dims(Y_denominator, -1))
        cos_theta = numerator / X_denominator / Y_denominator
        jitter = 1e-15
        theta = np.arccos(jitter + (1 - 2 * jitter) * cos_theta)

        K = self.variance * (1. / np.pi) * self._J(theta) \
            * X_denominator ** self.order \
            * Y_denominator ** self.order

        if eval_gradient:
            def f_arccos(theta):  # helper function
                return self.clone_with_theta(theta)(X, Y)

            return K, _approx_fprime(self.theta, f_arccos, 1e-10)
        else:
            return K

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : array, shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        X_product = self._weighted_product(X)
        theta = 0.0
        return self.variance * (1. / np.pi) * self._J(theta) * X_product ** self.order

    def is_stationary(self):
        """Returns whether the kernel is stationary. """
        return False

    def __repr__(self):
        return f"{self.__class__.__name__}(ord={int(self.order)}, var={self.variance:.3g}, " \
               f"w={self.weight_variances:.3g}, b={self.bias_variance:.3g})"


def create_target_function(list_of_target_functions, aggregation_function, inner_njobs=None, only_aggregate=False):
    """
    Create a global target function for the GP Regressor.
    Needs to take a list of individual target functions
    and a aggregation rule function.
    :param list_of_target_functions:
    :param aggregation_function:
    :param inner_njobs:
    :return: function which can take **kwargs and use
    these kwargs as input to all target_functions in
    list_of_target_functions and return their return
    values plus the aggregation_function of the return values
    """

    def return_function(uuid=0, best_score=999, **kwargs):
        """ runs all functions in list_of_target_functions with all values in kwarg and aggregates
        the results with aggregation_function
        :return: results with aggregate at -1"""
        test_array = list(kwargs.values())[0]
        if not hasattr(test_array, "shape") or len(test_array.shape) == 0:
            kwargs = {key: (lambda x: np.array(x).reshape(-1))(val) for key, val in kwargs.items()}
            test_array = list(kwargs.values())[0]

        # function_results = [func(**kwargs) for func in list_of_target_functions]
        if inner_njobs is None:
            n_jobs = 1
        elif inner_njobs == "auto":
            n_jobs = min(24, len(list_of_target_functions) * test_array.shape[0])
        else:
            n_jobs = inner_njobs
        unzipped_kwargs = [{key: val[i:i + 1] for key, val in kwargs.items()} for i in
                           range(test_array.shape[0])]
        zipped_list = [(func, kwarg) for kwarg in unzipped_kwargs for func in
                       list_of_target_functions]
        if n_jobs == 1:
            function_results = [func(uuid=uuid, best_score=best_score, **kwarg) for func, kwarg in zipped_list]
        else:
            function_results = Parallel(n_jobs=n_jobs, verbose=0, prefer="threads")(
                delayed(func)(uuid=uuid, best_score=best_score, **kwarg) for func, kwarg in zipped_list)

        ys = np.concatenate(function_results, axis=-1).reshape(test_array.shape[0], -1)
        res = np.concatenate([ys, aggregation_function(ys)], axis=1)
        if only_aggregate:
            res = np.concatenate([aggregation_function(ys), aggregation_function(ys)], axis=1)
        if test_array.shape[0] == 1:
            res = res[0]
        return res

    return return_function


def create_gp_list(json_dict, kernel=RationalQuadratic, **kwargs):
    gp_list = []
    gp_name_list = []
    for exp_name, exp_dict in sorted(json_dict.experiments.items()):
        number_of_scores = len(exp_dict.feature_score_names)
        # gps = [AnnModel(len(json_dict.parameters)) for _ in range(number_of_scores)]
        gps = [GaussianProcessRegressor(kernel=kernel(**kwargs), alpha=1e-5, normalize_y=False,
                                        n_restarts_optimizer=1)
               for _ in range(number_of_scores)]

        gp_list.extend(gps)
        gp_name_list.extend(exp_dict.feature_score_names)
    if json_dict.only_aggregate_score:
        gp_list = [GaussianProcessRegressor(kernel=kernel(**kwargs), alpha=1e-5, normalize_y=False,
                                            n_restarts_optimizer=1)]
        gp_name_list = ["aggregate"]

    return gp_list, gp_name_list


def create_kernel():
    # return ArcCosine(order=1) + RationalQuadratic(alpha=0.005)
    return 1.0 * RationalQuadratic(alpha=0.005)


def search_step(uuid, kappa, optimizer, target_function, time_df=None, json_dict=None,
                acq_warmup=1000, acq_n_iter=1, final=False, next_point=None):
    sum_starttime = datetime.now()
    suggest_duration = 0
    if next_point is None:
        utility = UtilityFunction(kind="ucb", kappa=kappa, xi=1e-2)
        suggest_starttime = datetime.now()
        next_point = optimizer.suggest(utility, n_iter=acq_n_iter, n_warmup=acq_warmup)
        suggest_duration = datetime.now() - suggest_starttime
    else:
        print(next_point)
    target_starttime = datetime.now()
    target = target_function(uuid=str(uuid), best_score=optimizer.min["target"],
                             **next_point)
    target_duration = datetime.now() - target_starttime
    previous_min = optimizer.min["target"]
    if target[-1] < previous_min and json_dict.create_plots:
        plot_sim(str(uuid), json_dict)
    optimizer.register(params=next_point, target=target)
    fitting_starttime = datetime.now()
    optimizer.fit_gp()
    fitting_duration = datetime.now() - fitting_starttime
    sum_duration = datetime.now() - sum_starttime
    summary = f"UUID: {uuid} - Kappa {str(np.round(kappa, 3)).ljust(4, ' ')}" \
              f" - Suggestion took {suggest_duration}, " \
              f"Target took {target_duration}, " \
              f"GP fitting took {fitting_duration}, Sum took {sum_duration}"
    if time_df is not None:
        time_df = pd.concat([time_df, pd.Series([uuid, datetime.now(), suggest_duration, target_duration,
                                                 fitting_duration],
                                                index=["k", "time", "suggest", "target", "fit"])],
                            ignore_index=True)
        save_time_df(time_df, json_dict)
    # print(summary)
    logger = getLogger(json_dict.resultsDir)
    logger.info(summary)
    if time_df is not None:
        return optimizer, time_df
    else:
        return optimizer

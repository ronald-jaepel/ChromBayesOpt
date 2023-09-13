import warnings

import numpy as np

from bayes_opt.event import Events, DEFAULT_EVENTS
from bayes_opt.bayesian_optimization import Observable, Queue
from bayes_opt.util import ensure_rng
from bayes_opt.logger import _get_default_logger
from .util import acq_min, create_sobol, UtilityFunction
from .target_space import TargetSpaceMulti


class BayesianOptimizationMulti(Observable):
    def __init__(self, f, pbounds, gp, random_state=None, verbose=2, n_warmup_points=None):
        """"""
        self._random_state = ensure_rng(random_state)

        # Data structure containing the function to be optimized, the bounds of
        # its domain, and a record of the evaluations we have done so far
        self._space = TargetSpaceMulti(f, pbounds,
                                       target_dims=gp.target_dims + 1,
                                       random_state=random_state,
                                       interpolate_nans=True)

        # queue
        self._queue = Queue()

        # Internal GP regressor
        self._gp = gp

        if n_warmup_points:
            bounds = np.array([val for key, val in sorted(pbounds.items())])
            self._warmup_points_for_acq_min = create_sobol(n_warmup_points, 0, bounds)
        else:
            self._warmup_points_for_acq_min = n_warmup_points

        self._verbose = verbose
        super().__init__(events=DEFAULT_EVENTS)
        self._prime_subscriptions()
        self.dispatch(Events.OPTIMIZATION_START)

    @property
    def space(self):
        return self._space

    @property
    def min(self):
        return self._space.min()

    @property
    def max(self):
        return self._space.max()

    @property
    def res(self):
        return self._space.res()

    def register(self, params, target):
        """Expect observation with known target"""
        if len(target.shape) == 1:
            self._space.register(params, target)
            self.dispatch(Events.OPTIMIZATION_STEP)

        else:
            for i in range(target.shape[0]):
                param_point = {key: val[i:i + 1] for key, val in params.items()}
                target_point = target[i, :]
                self._space.register(param_point, target_point)
                self.dispatch(Events.OPTIMIZATION_STEP)

    def probe(self, params, lazy=True):
        """Probe target of x"""
        if lazy:
            self._queue.add(params)
        else:
            self._space.probe(params)
            self.dispatch(Events.OPTIMIZATION_STEP)

    def fit_gp(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._space.params, self._space.component_target_space)

    def suggest(self, utility_function, n_iter=250, n_warmup=None):
        """Most promissing point to probe next"""
        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample())

        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._space.params, self._space.component_target_space)

        # Finding argmin of the acquisition function.
        suggestions = acq_min(
            ac=utility_function.utility,
            gp=self._gp,
            y_min=self._space.target.min(),
            x_min=self._space.params_to_array(self.min["params"]),
            bounds=self._space.bounds,
            random_state=self._random_state,
            n_warmup=n_warmup,
            n_iter=n_iter,
            warmup_points=self._warmup_points_for_acq_min,
            n_returns=n_iter
        )

        suggestions_filtered = [x for x in suggestions if not list(x) in self._space.params.tolist()]
        while len(suggestions_filtered) == 0:
            suggestions_changed = suggestions * (1 + (np.random.random(suggestions.size) - 0.5) * 0.001)
            for dimension in range(self._space.bounds.shape[0]):
                suggestions_changed[:, dimension] = min(max(suggestions_changed[:, dimension],
                                                            self._space.bounds[dimension, 0]),
                                                        self._space.bounds[dimension, 1])
            suggestions_filtered = [x for x in suggestions_changed if not list(x) in self._space.params.tolist()]

        # print(f"Suggested point {suggestions[0]} was not unique, changing it to {suggestions_filtered[0]}")

        suggestion = np.atleast_2d(suggestions_filtered[0])

        return self._space.array_to_params(suggestion)

    def _prime_queue(self, init_points):
        """Make sure there's something in the queue at the very beginning."""
        if self._queue.empty and self._space.empty:
            init_points = max(init_points, 1)

        for _ in range(init_points):
            self._queue.add(self._space.random_sample())

    def _prime_subscriptions(self):
        if not any([len(subs) for subs in self._events.values()]):
            _logger = _get_default_logger(self._verbose, is_constrained=False)
            self.subscribe(Events.OPTIMIZATION_START, _logger)
            self.subscribe(Events.OPTIMIZATION_STEP, _logger)
            self.subscribe(Events.OPTIMIZATION_END, _logger)

    def minimize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ucb',
                 kappa=2.576,
                 xi=0.0,
                 **gp_params):
        """Mazimize your function"""
        self._prime_subscriptions()
        self.dispatch(Events.OPTIMIZATION_END)
        self._prime_queue(init_points)
        self.set_gp_params(**gp_params)

        util = UtilityFunction(kind=acq, kappa=kappa, xi=xi)
        iteration = 0
        while not self._queue.empty or iteration < n_iter:
            try:
                x_probe = next(self._queue)
            except StopIteration:
                x_probe = self.suggest(util)
                iteration += 1

            self.probe(x_probe, lazy=False)

        self.dispatch(Events.OPTIMIZATION_END)

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        Parameters
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds
        """
        self._space.set_bounds(new_bounds)

    def set_gp_params(self, **params):
        self._gp.set_params(**params)

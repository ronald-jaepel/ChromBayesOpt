from copy import copy

import numpy as np
from scipy.interpolate import NearestNDInterpolator

from bayes_opt.util import ensure_rng
from bayes_opt.target_space import TargetSpace, _hashable


class TargetSpaceMulti(TargetSpace):
    """
    Holds the param-space coordinates (X) and target values (Y)
    Allows for constant-time appends while ensuring no duplicates are added

    Example
    -------
    # def target_func(p1, p2):
    #     return p1 + p2
    # pbounds = {'p1': (0, 1), 'p2': (1, 100)}
    # space = TargetSpace(target_func, pbounds, random_state=0)
    # x = space.random_points(1)[0]
    # y = space.register_point(x)
    # assert self.min_point()['min_val'] == y
    """

    def __init__(self, target_func, pbounds, target_dims, random_state=None,
                 interpolate_nans=False):
        """
        Parameters
        ----------
        target_func : function
            Function to be minimized.

        pbounds : dict
            Dictionary with parameters names as keys and a tuple with minimum
            and minimum values.

        random_state : int, RandomState, or None
            optionally specify a seed for a random number generator

        interpolate_nans : bool, if Nan values in the space should be
            interpolated using nearest neighbour interpolation
        """
        self.interpolate_nans = interpolate_nans
        self.random_state = ensure_rng(random_state)

        # The function to be optimized
        self.target_func = target_func
        self.target_dims = target_dims

        # Get the name of the parameters
        self._keys = sorted(pbounds)
        # Create an array with parameters bounds
        self._bounds = np.array(
            [item[1] for item in sorted(pbounds.items(), key=lambda x: x[0])],
            dtype=np.float
        )

        # preallocated memory for X and Y points
        self._params = np.empty(shape=(0, self.dim))
        self._target = np.empty(shape=(0, self.target_dims))

        # keep track of unique points we have seen so far
        self._cache = {}

    def interpolate_dim(self, dim_index, use_extremes=True):
        mask = np.isnan(self._target[:, dim_index])
        given_y = self.raw_target[~mask, dim_index]
        given_x = self._params[~mask, :]
        target_x = self._params[mask, :]
        target_y = copy(self.raw_target[:, dim_index])
        if use_extremes:
            y_interpol = np.sign(NearestNDInterpolator(given_x, given_y)(target_x))
            y_interpol[y_interpol < 0] = given_y.min()
            y_interpol[y_interpol > 0] = given_y.max()
        else:
            y_interpol = NearestNDInterpolator(given_x, given_y)(target_x)
        target_y[mask] = y_interpol
        return target_y

    @property
    def full_target_space(self):
        if self.interpolate_nans and np.isnan(self._target).any():
            targets = []
            for dim in range(self.target_dims - 1):
                targets.append(self.interpolate_dim(dim))
            targets.append(self.interpolate_dim(-1, use_extremes=False))
            target = np.stack(targets).T
        else:
            target = self._target
        return target

    @property
    def component_target_space(self):
        if self.interpolate_nans and np.isnan(self._target).any():
            targets = []
            for dim in range(self.target_dims - 1):
                targets.append(self.interpolate_dim(dim))
            target = np.stack(targets).T
        else:
            target = self._target[:, :-1]
        return target

    @property
    def target(self):
        return self._target[:, -1]

    @property
    def raw_target(self):
        return self._target

    def array_to_params(self, x):
        try:
            assert x.shape[1] == len(self.keys)
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(len(self.keys))
            )
        return dict(zip(self.keys, x.T))

    def register(self, params, target):
        """
        Append a point and its target value to the known data.

        Parameters
        ----------
        params : ndarray
            a single point, with len(x) == self.dim

        target : ndarray
            target function value

        Raises
        ------
        KeyError:
            if the point is not unique

        Notes
        -----
        runs in ammortized constant time

        Example
        -------
        # pbounds = {'p1': (0, 1), 'p2': (1, 100)}
        # space = TargetSpace(lambda p1, p2: p1 + p2, pbounds)
        # len(space)
        0
        # x = np.array([0, 0])
        # y = 1
        # space.add_observation(x, y)
        # len(space)
        1
        """
        x = self._as_array(params)
        if x in self:
            print(f'Data point {x} is not unique')
            return

        # Insert data into unique dictionary
        self._cache[_hashable(x.ravel())] = target

        self._params = np.concatenate([self._params, x.reshape(1, -1)])
        self._target = np.concatenate([self._target, target.reshape(1, -1)])

    def min(self):
        """Get minimum target value found and corresponding parametes."""
        try:
            res = {
                'target': np.nanmin(self.target),
                'params': dict(
                    zip(self.keys, self.params[np.nanargmin(self.target)])
                )
            }
        except ValueError:
            res = {}
        return res

    def max(self):
        """To allow integration with max-oriented programs.
        Get negative min target value found and corresponding parametes."""
        try:
            res = {
                'target': -np.nanmin(self.target),
                'params': dict(
                    zip(self.keys, self.params[np.nanargmin(self.target)])
                )
            }
        except ValueError:
            res = {}
        return res
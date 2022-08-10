import numpy as np

transforms_dict = {'log': lambda x: np.power(10.0, x), "lin": lambda x: x,
                   "log_inv": lambda x: np.log10(x), "lin_inv": lambda x: x}


def mean_agg_func(ys, std_devs=None):
    ys = np.array(ys)
    # smooth approximation for |ys|
    y_scores_absolute = ys ** 2 / np.sqrt(ys ** 2 + 1e-8)
    if std_devs is None:
        y_mean = np.mean(y_scores_absolute, axis=1).reshape(-1, 1)
    else:
        weights = 1 / (np.square(std_devs) + 1)
        y_scores_weighted = y_scores_absolute * weights / np.sum(weights)
        y_mean = np.mean(y_scores_weighted, axis=1).reshape(-1, 1)
    return y_mean


def var_agg_func(ys, std_devs):
    std_devs = np.array(std_devs)
    std_dev = np.sqrt(np.sum(std_devs ** 2, axis=1)) / ys.shape[1]
    return std_dev.reshape(-1, 1)

from copy import copy

import numpy as np
from addict import Dict
from numpy import trapz
from scipy import signal
from scipy import stats
from scipy.interpolate import InterpolatedUnivariateSpline, PchipInterpolator
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from scipy.special import erfcx

"""scores"""


def range_function(x):
    return x / (1 + np.abs(x))


def create_ranged_function(func):
    def ranged_func(*args, **kwargs):
        un_ranged_results = func(*args, **kwargs)
        ranged_results = list(range_function(np.array(un_ranged_results)))
        return ranged_results

    return ranged_func


def roll(x, shift):
    if shift > 0:
        temp = np.pad(x, (shift, 0), mode='reflect', reflect_type='odd')
        return temp[:-shift]
    elif shift < 0:
        temp = np.pad(x, (0, np.abs(shift)), mode='reflect', reflect_type='odd')
        return temp[np.abs(shift):]
    else:
        return x


def find_peak(times, data):
    """Return tuples of (times,data) for the peak we need"""
    min_idx = np.argmin(data)
    max_idx = np.argmax(data)
    return (times[max_idx], data[max_idx]), (times[min_idx], data[min_idx])


def pearson_spline2(exp_time_values, exp_data_values, sim_time_values, sim_data_values,
                    time_selection):
    def pear_corr(cr):
        # handle the case where a nan is returned
        if np.isnan(cr):
            return 0.0
        if cr < 0.5:
            out = 1.0 / 3.0 * cr + 1.0 / 3.0
        else:
            out = cr
        return out

    # sim_data_values_untrimmed = copy(sim_data_values)
    # exp_time_values_untrimmed = copy(exp_time_values)
    exp_time_values, exp_data_values, sim_time_values, sim_data_values = trim_to_time_selection(
        exp_time_values, exp_data_values, sim_time_values, sim_data_values, time_selection)

    spline_sim = InterpolatedUnivariateSpline(x=sim_time_values, y=sim_data_values, ext='zeros')
    spline_exp = InterpolatedUnivariateSpline(x=exp_time_values, y=exp_data_values, ext='zeros')

    exp_time_values_ext = np.array(
        np.linspace(exp_time_values[0], exp_time_values[-1], len(exp_time_values) * 100))
    sim_data_values_ext = spline_sim(exp_time_values_ext)
    exp_data_values_ext = spline_exp(exp_time_values_ext)

    corr = signal.correlate(exp_data_values_ext, sim_data_values_ext) / (
            np.linalg.norm(sim_data_values_ext) * np.linalg.norm(exp_data_values_ext))

    index = np.argmax(corr) + 1
    roll_index = index - len(exp_time_values_ext)
    sim_time_values_rolled = roll(exp_time_values_ext, shift=int(np.ceil(roll_index)))
    sim_data_values_rolled = roll(sim_data_values_ext, shift=int(np.ceil(roll_index)))

    time_difference = exp_time_values_ext[int(len(exp_time_values_ext) / 2)] \
                      - sim_time_values_rolled[int(len(exp_time_values_ext) / 2)]

    pearson_score = pearsonr(sim_data_values_rolled, exp_data_values_ext)

    return time_difference, pear_corr(pearson_score[0])


def pearson_transform(pearson_score):
    pearson_score = np.power(10., 2 * (pearson_score - 1))
    return pearson_score


def diff_time_tolerance_transform(diff_time, time_tolerance, attenuation_factor):
    if time_tolerance > 0:
        if np.abs(diff_time) < time_tolerance:
            diff_time = diff_time / attenuation_factor
        else:
            diff_time = diff_time - np.sign(diff_time) * time_tolerance * (
                    1 - 1 / attenuation_factor)
    return diff_time


def exponnorm2(x, h=1, m=1, s=1, t=1):
    y = h * np.exp(-0.5 * ((x - m) / s) ** 2) * s / t * np.sqrt(np.pi / 2) \
        * erfcx(1 / np.sqrt(2) * (s / t - ((x - m) / s)))
    y[~np.isfinite(y)] = 0
    return y


def fit_emg(x, y):
    x_where_max_y = x[np.argmax(y)]
    fitting_x = (x - x_where_max_y) / 100
    initial_guess = (y.max(), 0, 0.01, 0.8)
    initial_guess2 = (y.max(), 0, 0.01, 10)
    failed_attempts = 0
    try:
        popt1, _ = curve_fit(exponnorm2, fitting_x, y, p0=initial_guess)
    except RuntimeError as ex:
        popt1 = initial_guess
        failed_attempts += 1
        print("attempt 1 failed")
    try:
        popt2, _ = curve_fit(exponnorm2, fitting_x, y, p0=initial_guess2)
    except RuntimeError as ex:
        popt2 = initial_guess2
        print("attempt 2 failed")
        if failed_attempts:
            print("both failed")

    calc1 = exponnorm2(fitting_x, *popt1)
    calc2 = exponnorm2(fitting_x, *popt2)
    sse1 = sum(i * i for i in [i - j for i, j in zip(y, calc1)])
    sse2 = sum(i * i for i in [i - j for i, j in zip(y, calc2)])
    if sse1 < sse2:
        popt = popt1
        # print("choosing pop1")
    else:
        popt = popt2
        # print("choosing pop2")
    popt = [popt[0], popt[1] + x_where_max_y, popt[2], popt[3]]
    return popt


def trim_to_time_selection(exp_time_values, exp_data_values, sim_time_values, sim_data_values,
                           time_selection):
    if time_selection is not None and time_selection is not False:
        exp_data_values = exp_data_values[(time_selection[0] <= exp_time_values) &
                                          (exp_time_values <= time_selection[1])]
        exp_time_values = exp_time_values[(time_selection[0] <= exp_time_values) &
                                          (exp_time_values <= time_selection[1])]
        sim_data_values = sim_data_values[(time_selection[0] <= sim_time_values) &
                                          (sim_time_values <= time_selection[1])]
        sim_time_values = sim_time_values[(time_selection[0] <= sim_time_values) &
                                          (sim_time_values <= time_selection[1])]
    return exp_time_values, exp_data_values, sim_time_values, sim_data_values


def emg_score(exp_time_values, exp_data_values, sim_time_values, sim_data_values,
              time_selection, meta_info):
    exp_time_values, exp_data_values, sim_time_values, sim_data_values = trim_to_time_selection(
        exp_time_values, exp_data_values, sim_time_values, sim_data_values, time_selection)

    exp_emg = fit_emg(exp_time_values, exp_data_values)
    sim_emg = fit_emg(sim_time_values, sim_data_values)

    time_score = (exp_emg[1] - sim_emg[1]) / exp_emg[1] * 4
    width_score = (exp_emg[2] - sim_emg[2]) / exp_emg[2]
    return time_score, width_score


def get_time_and_shape_score(exp_time_values, exp_data_values, sim_time_values, sim_data_values,
                             time_selection, meta_info):
    exp_time_values_trim, exp_data_values_trim, sim_time_values_trim, sim_data_values_trim \
        = trim_to_time_selection(exp_time_values, exp_data_values, sim_time_values,
                                 sim_data_values, time_selection)

    total_exp_area = trapz(y=exp_data_values, x=exp_time_values)
    if time_selection is not None and time_selection is not False and time_selection[0] != 0:
        sim_data_values_pre = sim_data_values[time_selection[0] > sim_time_values]
        sim_time_values_pre = sim_time_values[time_selection[0] > sim_time_values]
        pre_gradient_sim_area = trapz(y=sim_data_values_pre, x=sim_time_values_pre)
        fraction_pre_elution = pre_gradient_sim_area / total_exp_area
        in_gradient_sim_area = trapz(y=sim_data_values_trim, x=sim_time_values_trim)
    else:
        fraction_pre_elution = 0
        in_gradient_sim_area = trapz(y=sim_data_values, x=sim_time_values)
    # in_gradient_exp_area = trapz(y=exp_data_values_trim, x=exp_time_values_trim)
    fraction_in_elution = in_gradient_sim_area / total_exp_area

    max_peak_exp = find_peak(exp_time_values_trim, exp_data_values_trim)[0]

    # if more than 95% didn't bind: treat it as non-binding
    if fraction_pre_elution > 0.95:
        time_selection = [0, 1e12]
        diff_time, pearson_score = pearson_spline2(exp_time_values, exp_data_values,
                                                   sim_time_values, sim_data_values, time_selection)
    # if less than 50% didn't bind and nothing eluted during the gradient: treat it as strongly-binding but slow
    elif fraction_pre_elution < 0.5 and fraction_in_elution < 0.05:
        diff_time = max_peak_exp[0] - exp_time_values[-1]
        pearson_score = 0
    else:
        diff_time, pearson_score = pearson_spline2(exp_time_values, exp_data_values,
                                                   sim_time_values, sim_data_values, time_selection)

    time_score = - diff_time / max_peak_exp[0]

    pearson_score = pearson_transform(pearson_score)

    return time_score, 1 - pearson_score


def get_shape_score(exp_time_values, exp_data_values, sim_time_values, sim_data_values,
                    time_selection, meta_info):
    return \
        get_time_and_shape_score(exp_time_values, exp_data_values, sim_time_values, sim_data_values,
                                 time_selection, meta_info)[-2:-1]


def get_time_score(exp_time_values, exp_data_values, sim_time_values, sim_data_values,
                   time_selection, meta_info):
    return \
        get_time_and_shape_score(exp_time_values, exp_data_values, sim_time_values, sim_data_values,
                                 time_selection, meta_info)[0:1]


def calculate_height_difference(exp_time_values, exp_data_values, sim_time_values, sim_data_values,
                                time_selection, meta_info):
    # spline based interpolation
    spline_sim = InterpolatedUnivariateSpline(x=sim_time_values, y=sim_data_values, ext='zeros')
    spline_exp = InterpolatedUnivariateSpline(x=exp_time_values, y=exp_data_values, ext='zeros')

    exp_time_values_ext = np.array(
        np.linspace(exp_time_values[0], exp_time_values[-1], len(exp_time_values) * 1000))
    sim_data_values_ext = spline_sim(exp_time_values_ext)
    exp_data_values_ext = spline_exp(exp_time_values_ext)

    # restrict to time selection
    exp_selected = exp_data_values_ext[
        (time_selection[0] < exp_time_values_ext) & (exp_time_values_ext < time_selection[1])]
    sim_selected = sim_data_values_ext[
        (time_selection[0] < exp_time_values_ext) & (exp_time_values_ext < time_selection[1])]
    return exp_selected.max(), sim_selected.max()


def calculate_height_difference_area_normed(exp_time_values, exp_data_values, sim_time_values,
                                            sim_data_values, time_selection, meta_info):
    exp_time_values_t, exp_data_values_t, sim_time_values_t, sim_data_values_t \
        = trim_to_time_selection(exp_time_values, exp_data_values, sim_time_values,
                                 sim_data_values, time_selection)

    if time_selection is not None and time_selection is not False:
        sim_data_values_pre = sim_data_values[time_selection[0] > sim_time_values]
        sim_time_values_pre = sim_time_values[time_selection[0] > sim_time_values]
        pre_gradient_sim_area = trapz(y=sim_data_values_pre, x=sim_time_values_pre)
    else:
        pre_gradient_sim_area = 0
    total_exp_area = trapz(y=exp_data_values, x=exp_time_values)
    in_gradient_sim_area = trapz(y=sim_data_values_t, x=sim_time_values_t)
    in_gradient_exp_area = trapz(y=exp_data_values_t, x=exp_time_values_t)
    fraction_pre_elution = pre_gradient_sim_area / total_exp_area
    fraction_in_elution = in_gradient_sim_area / total_exp_area

    if fraction_pre_elution < 0.96 and fraction_in_elution > 0.04:
        # area normalization
        sim_data_values_t = sim_data_values_t / in_gradient_sim_area * in_gradient_exp_area

    # spline based interpolation
    spline_sim = InterpolatedUnivariateSpline(x=sim_time_values_t, y=sim_data_values_t, ext='zeros')
    spline_exp = InterpolatedUnivariateSpline(x=exp_time_values_t, y=exp_data_values_t, ext='zeros')

    exp_time_values_ext = np.array(
        np.linspace(exp_time_values_t[0], exp_time_values_t[-1], len(exp_time_values) * 100))
    sim_data_values_ext = spline_sim(exp_time_values_ext)
    exp_data_values_ext = spline_exp(exp_time_values_ext)

    return exp_data_values_ext.max(), sim_data_values_ext.max()


def calculate_height_difference_area_normed_spline(exp_time_values, exp_data_values, sim_time_values,
                                                   sim_data_values, time_selection, meta_info):
    exp_time_values, exp_data_values, sim_time_values, sim_data_values \
        = trim_to_time_selection(exp_time_values, exp_data_values, sim_time_values,
                                 sim_data_values, time_selection)

    # spline based interpolation
    spline_sim = InterpolatedUnivariateSpline(x=sim_time_values, y=sim_data_values, ext='zeros')
    spline_exp = InterpolatedUnivariateSpline(x=exp_time_values, y=exp_data_values, ext='zeros')

    exp_time_values = np.array(
        np.linspace(exp_time_values[0], exp_time_values[-1], len(exp_time_values) * 100))
    sim_data_values = spline_sim(exp_time_values)
    exp_data_values = spline_exp(exp_time_values)

    if time_selection is not None and time_selection is not False:
        sim_data_values_pre = sim_data_values[time_selection[0] > sim_time_values]
        sim_time_values_pre = sim_time_values[time_selection[0] > sim_time_values]
        pre_gradient_sim_area = trapz(y=sim_data_values_pre, x=sim_time_values_pre)
    else:
        pre_gradient_sim_area = 0
    total_exp_area = trapz(y=exp_data_values, x=exp_time_values)
    in_gradient_sim_area = trapz(y=sim_data_values, x=exp_time_values)
    in_gradient_exp_area = trapz(y=exp_data_values, x=exp_time_values)
    fraction_pre_elution = pre_gradient_sim_area / total_exp_area
    fraction_in_elution = in_gradient_sim_area / total_exp_area

    if fraction_pre_elution < 0.96 and fraction_in_elution > 0.04:
        # area normalization
        sim_data_values = sim_data_values / in_gradient_sim_area * in_gradient_exp_area

    return exp_data_values.max(), sim_data_values.max()


def get_peak_height_score(exp_time_values, exp_data_values, sim_time_values, sim_data_values,
                          time_selection, meta_info):
    exp_max, sim_max = calculate_height_difference(exp_time_values, exp_data_values,
                                                   sim_time_values, sim_data_values,
                                                   time_selection, meta_info)

    # peak height score calculation
    peak_height_score = (exp_max - sim_max) / exp_max
    return [peak_height_score]


def get_peak_height_log_score(exp_time_values, exp_data_values, sim_time_values, sim_data_values,
                              time_selection, meta_info):
    def peak_height_function(x):
        if x < -1.5:
            x = -1.5
        return x

    exp_max, sim_max = calculate_height_difference_area_normed(exp_time_values, exp_data_values,
                                                               sim_time_values, sim_data_values,
                                                               time_selection, meta_info)

    # peak height score calculation
    peak_height_score = np.log2(sim_max / exp_max)
    peak_height_score = peak_height_function(peak_height_score)
    return [peak_height_score]


def get_peak_height_log_spline_score(exp_time_values, exp_data_values, sim_time_values, sim_data_values,
                                     time_selection, meta_info):
    def peak_height_function(x):
        if x < -1.5:
            x = -1.5
        return x

    exp_max, sim_max = calculate_height_difference_area_normed_spline(exp_time_values, exp_data_values,
                                                                      sim_time_values, sim_data_values,
                                                                      time_selection, meta_info)

    # peak height score calculation
    peak_height_score = np.log2(sim_max / exp_max)
    peak_height_score = peak_height_function(peak_height_score)
    return [peak_height_score]


def get_skewness_score(exp_time_values, exp_data_values, sim_time_values, sim_data_values,
                       time_selection, meta_info):
    exp_data_values[exp_data_values <= 0] = 0
    sum_of_exp_data = np.sum(exp_data_values)
    exp_data_normalized = exp_data_values / sum_of_exp_data
    rv_exp = stats.rv_discrete(name="custom", values=(exp_time_values, exp_data_normalized))
    mean_exp = rv_exp.mean()
    median_exp = rv_exp.median()
    std_exp = rv_exp.std()
    skewness_exp = (mean_exp - median_exp) / std_exp

    sim_data_values[sim_data_values <= 0] = 0
    sum_of_sim_data = np.sum(sim_data_values)
    sim_data_normalized = sim_data_values / sum_of_sim_data
    rv_sim = stats.rv_discrete(name="custom", values=(exp_time_values, sim_data_normalized))
    mean_sim = rv_sim.mean()
    median_sim = rv_sim.median()
    std_sim = rv_sim.std()
    skewness_sim = (mean_sim - median_sim) / std_sim
    score = skewness_sim - skewness_exp
    # if np.isnan(score):
    #     score = -10
    return [score]


def calculate_distribution_skew(data, time):
    data[time <= 0] = 0
    data[data <= 0] = 0
    sum_of_data = np.sum(data)
    sim_data_normalized = data / sum_of_data
    rv = stats.rv_discrete(name="custom", values=(time, sim_data_normalized))
    mean = rv.mean()
    median = rv.median()
    std = rv.std()
    skewness = (mean - median) / std
    return skewness


def get_skewness_spline_score(exp_time_values, exp_data_values, sim_time_values, sim_data_values,
                              time_selection, meta_info):
    spline_sim = InterpolatedUnivariateSpline(x=sim_time_values, y=sim_data_values, ext='zeros')
    spline_exp = InterpolatedUnivariateSpline(x=exp_time_values, y=exp_data_values, ext='zeros')

    exp_time_values_ext = np.array(
        np.linspace(exp_time_values[0], exp_time_values[-1], len(exp_time_values) * 100))
    exp_time_values_ext = exp_time_values_ext[exp_time_values_ext >= 0]
    sim_data_values_ext = spline_sim(exp_time_values_ext)
    exp_data_values_ext = spline_exp(exp_time_values_ext)

    sim_data_values_ext = sim_data_values_ext \
                          / trapz(y=sim_data_values_ext, x=exp_time_values_ext) \
                          * trapz(y=exp_data_values_ext, x=exp_time_values_ext)

    # restrict to time selection
    exp_selected = exp_data_values_ext[
        (time_selection[0] < exp_time_values_ext) & (exp_time_values_ext < time_selection[1])]
    sim_selected = sim_data_values_ext[
        (time_selection[0] < exp_time_values_ext) & (exp_time_values_ext < time_selection[1])]
    time_selected = exp_time_values_ext[
        (time_selection[0] < exp_time_values_ext) & (exp_time_values_ext < time_selection[1])]

    skewness_sim = calculate_distribution_skew(sim_selected, time_selected)
    skewness_exp = calculate_distribution_skew(exp_selected, time_selected)
    score = skewness_sim - skewness_exp
    return [score]


def calculate_skewness_10percent(data, time):
    data_max = data.max()
    index_max = data.argmax()
    onset_10percent = (data >= data_max * 0.1).argmax()
    offset_10percent = len(data) - 1 - (data >= data_max / 10)[::-1].argmax()
    skewness = (time[offset_10percent] - time[index_max]) / (time[index_max] - time[onset_10percent])
    return skewness


def get_skewness_10percent_score(exp_time_values, exp_data_values, sim_time_values, sim_data_values,
                                 time_selection, meta_info):
    spline_sim = InterpolatedUnivariateSpline(x=sim_time_values, y=sim_data_values, ext='zeros')
    spline_exp = InterpolatedUnivariateSpline(x=exp_time_values, y=exp_data_values, ext='zeros')

    exp_time_values_ext = np.array(
        np.linspace(exp_time_values[0], exp_time_values[-1], len(exp_time_values) * 100))
    exp_time_values_ext = exp_time_values_ext[exp_time_values_ext >= 0]
    sim_data_values_ext = spline_sim(exp_time_values_ext)
    exp_data_values_ext = spline_exp(exp_time_values_ext)

    # restrict to time selection
    exp_selected = exp_data_values_ext[
        (time_selection[0] < exp_time_values_ext) & (exp_time_values_ext < time_selection[1])]
    sim_selected = sim_data_values_ext[
        (time_selection[0] < exp_time_values_ext) & (exp_time_values_ext < time_selection[1])]
    time_selected = exp_time_values_ext[
        (time_selection[0] < exp_time_values_ext) & (exp_time_values_ext < time_selection[1])]

    skewness_exp = calculate_skewness_10percent(exp_selected, time_selected)
    skewness_sim = calculate_skewness_10percent(sim_selected, time_selected)

    score = (skewness_sim - skewness_exp) / skewness_exp
    return [score]


def bt_score_nans(exp_time_values, exp_data_values, sim_time_values, sim_data_values,
                  time_selection, meta_info):
    exp_time_values, exp_data_values, sim_time_values, sim_data_values = trim_to_time_selection(
        exp_time_values, exp_data_values, sim_time_values, sim_data_values, time_selection)

    twenty_perc_exp = np.where(exp_data_values > 0.2)[0]
    twenty_perc_sim = np.where(sim_data_values > 0.2)[0]
    ninety_perc_exp = np.where(exp_data_values > 0.9)[0]
    ninety_perc_sim = np.where(sim_data_values > 0.9)[0]

    if len(twenty_perc_sim) < 2:
        twenty_score_onset = np.nan
        twenty_score_offset = np.nan
    else:
        twenty_score_onset = (exp_time_values[twenty_perc_exp[0]]
                              - sim_time_values[twenty_perc_sim[0]]) \
                             / exp_time_values[twenty_perc_exp[0]]
        twenty_score_offset = (exp_time_values[twenty_perc_exp[-1]]
                               - sim_time_values[twenty_perc_sim[-1]]) \
                              / exp_time_values[twenty_perc_exp[-1]]
        twenty_score_onset *= 1
        twenty_score_offset *= 10

    if len(ninety_perc_sim) < 2:
        ninety_score_onset = np.nan
        ninety_score_offset = np.nan
    else:
        ninety_score_onset = (exp_time_values[ninety_perc_exp[0]]
                              - sim_time_values[ninety_perc_sim[0]]) \
                             / exp_time_values[ninety_perc_exp[0]]
        ninety_score_offset = (exp_time_values[ninety_perc_exp[-1]]
                               - sim_time_values[ninety_perc_sim[-1]]) \
                              / exp_time_values[ninety_perc_exp[-1]]
        ninety_score_onset *= 2
        ninety_score_offset *= 20

    return [twenty_score_onset, twenty_score_offset, ninety_score_onset, ninety_score_offset]


def bt_score(exp_time_values, exp_data_values, sim_time_values, sim_data_values,
             time_selection, meta_info):
    """
    Uses max of experimental data to set thresholds
    -> requires experimental data to reach full elution conc.
    :param exp_time_values:
    :param exp_data_values:
    :param sim_time_values:
    :param sim_data_values:
    :param time_selection:
    :param meta_info:
    :return lower_onset_score, onset_delay_score, max_value_score:
    """
    exp_time_values, exp_data_values, sim_time_values, sim_data_values = trim_to_time_selection(
        exp_time_values, exp_data_values, sim_time_values, sim_data_values, time_selection)

    spline_sim = PchipInterpolator(x=sim_time_values, y=sim_data_values)
    spline_exp = PchipInterpolator(x=exp_time_values, y=exp_data_values)

    exp_time_values = np.array(
        np.linspace(exp_time_values[0], exp_time_values[-1], len(exp_time_values) * 100))
    sim_time_values = exp_time_values
    sim_data_values = spline_sim(exp_time_values)
    exp_data_values = spline_exp(exp_time_values)

    max_exp_position = exp_data_values.argmax()
    max_exp_value = exp_data_values[max_exp_position]
    max_sim_position = sim_data_values.argmax()
    max_sim_value = sim_data_values[max_sim_position]

    if max_sim_value < max_exp_value / 100:
        return [np.nan, np.nan, 1]

    max_value_score = (max_exp_value - max_sim_value) / max_exp_value  # percentage value difference

    middle_thr_exp = np.where(exp_data_values > 0.5 * max_exp_value)[0]
    middle_thr_sim = np.where(sim_data_values > 0.5 * max_sim_value)[0]
    upper_thr_exp = np.where(exp_data_values > 0.98 * max_exp_value)[0]
    upper_thr_sim = np.where(sim_data_values > 0.98 * max_sim_value)[0]

    if len(middle_thr_sim) <= 1:
        lower_onset_score = np.nan if meta_info["using_nans"] else -1
    else:
        onset_time_exp = exp_time_values[middle_thr_exp[0]]
        onset_time_sim = sim_time_values[middle_thr_sim[0]]
        lower_onset_score = (onset_time_exp - onset_time_sim) / onset_time_exp

    if len(upper_thr_sim) <= 1:
        onset_delay_score = np.nan if meta_info["using_nans"] else -1
    else:
        onset_delay_exp = exp_time_values[upper_thr_exp[0]] - exp_time_values[middle_thr_exp[0]]
        onset_delay_sim = sim_time_values[upper_thr_sim[0]] - sim_time_values[middle_thr_sim[0]]
        onset_delay_score = (onset_delay_exp - onset_delay_sim) / onset_delay_exp

    return [lower_onset_score, onset_delay_score, max_value_score]


def bt_score2(exp_time_values, exp_data_values, sim_time_values, sim_data_values,
              time_selection, meta_info):
    """
    Uses max of experimental data to set thresholds
    -> requires experimental data to reach full elution conc.
    :param exp_time_values:
    :param exp_data_values:
    :param sim_time_values:
    :param sim_data_values:
    :param time_selection:
    :param meta_info:
    :return mid_onset_score, lower_lead_score, onset_delay_score, max_value_score:
    """
    exp_time_values, exp_data_values, sim_time_values, sim_data_values = trim_to_time_selection(
        exp_time_values, exp_data_values, sim_time_values, sim_data_values, time_selection)

    spline_sim = PchipInterpolator(x=sim_time_values, y=sim_data_values)
    spline_exp = PchipInterpolator(x=exp_time_values, y=exp_data_values)

    exp_time_values = np.array(
        np.linspace(exp_time_values[0], exp_time_values[-1], len(exp_time_values) * 100))
    sim_time_values = exp_time_values
    sim_data_values = spline_sim(exp_time_values)
    exp_data_values = spline_exp(exp_time_values)

    max_exp_position = exp_data_values.argmax()
    max_exp_value = exp_data_values[max_exp_position]
    max_sim_position = sim_data_values.argmax()
    max_sim_value = sim_data_values[max_sim_position]

    if max_sim_value < max_exp_value / 100:
        return [np.nan, np.nan, np.nan, 1]

    max_value_score = (max_exp_value - max_sim_value) / max_exp_value  # percentage value difference

    middle_thr_exp = np.where(exp_data_values > 0.5 * max_exp_value)[0]
    middle_thr_sim = np.where(sim_data_values > 0.5 * max_sim_value)[0]
    upper_thr_exp = np.where(exp_data_values > 0.95 * max_exp_value)[0]
    upper_thr_sim = np.where(sim_data_values > 0.95 * max_sim_value)[0]
    lower_thr_exp = np.where(exp_data_values > 0.85 * max_exp_value)[0]
    lower_thr_sim = np.where(sim_data_values > 0.85 * max_sim_value)[0]

    if len(middle_thr_sim) <= 1:
        middle_onset_score = np.nan if meta_info["using_nans"] else -1
    else:
        onset_time_exp = exp_time_values[middle_thr_exp[0]]
        onset_time_sim = sim_time_values[middle_thr_sim[0]]
        middle_onset_score = (onset_time_exp - onset_time_sim) / onset_time_exp

    if len(upper_thr_sim) <= 1:
        onset_delay_score = np.nan if meta_info["using_nans"] else -1
    else:
        onset_delay_exp = exp_time_values[upper_thr_exp[0]] - exp_time_values[middle_thr_exp[0]]
        onset_delay_sim = sim_time_values[upper_thr_sim[0]] - sim_time_values[middle_thr_sim[0]]
        onset_delay_score = (onset_delay_exp - onset_delay_sim) / onset_delay_exp

    if len(lower_thr_sim) <= 1:
        lower_lead_score = np.nan if meta_info["using_nans"] else -1
    else:
        onsetlower_delay_exp = exp_time_values[lower_thr_exp[0]] - exp_time_values[middle_thr_exp[0]]
        onsetlower_delay_sim = sim_time_values[lower_thr_sim[0]] - sim_time_values[middle_thr_sim[0]]
        lower_lead_score = (onsetlower_delay_exp - onsetlower_delay_sim) / onsetlower_delay_exp

    return [middle_onset_score, lower_lead_score, onset_delay_score, max_value_score]


def pre_gradient_elution(exp_time_values, exp_data_values, sim_time_values, sim_data_values,
                         time_selection, meta_info):
    exp_data_values = exp_data_values[time_selection[0] > exp_time_values]
    exp_time_values = exp_time_values[time_selection[0] > exp_time_values]
    sim_data_values = sim_data_values[time_selection[0] > sim_time_values]
    sim_time_values = sim_time_values[time_selection[0] > sim_time_values]
    pre_gradient_exp = trapz(y=exp_data_values, x=exp_time_values)
    pre_gradient_sim = trapz(y=sim_data_values, x=sim_time_values)
    delta = pre_gradient_sim - pre_gradient_exp
    return delta


def get_sse_nonpen_score(exp_time_values: np.ndarray, exp_data_values: np.ndarray,
                         sim_time_values: np.ndarray, sim_data_values: np.ndarray,
                         time_selection: tuple, meta_info: dict):
    exp_max_value = exp_data_values.max()
    fifty_percent_index = np.where(exp_data_values > 0.5 * exp_max_value)[0][0]
    fifty_percent_time = exp_time_values[fifty_percent_index]
    time_selection = (0, fifty_percent_time)
    sse_score = get_sse_score(exp_time_values, exp_data_values, sim_time_values, sim_data_values, time_selection,
                              meta_info)[0]
    return [sse_score, ]


def get_sse_score(exp_time_values: np.ndarray, exp_data_values, sim_time_values: np.ndarray, sim_data_values,
                  time_selection, meta_info):
    exp_time_values, exp_data_values, sim_time_values, sim_data_values = trim_to_time_selection(
        exp_time_values, exp_data_values, sim_time_values, sim_data_values, time_selection)
    if not all(exp_time_values == sim_time_values):
        print("interpolating sse")
        spline_sim = InterpolatedUnivariateSpline(x=sim_time_values, y=sim_data_values, ext='zeros')
        sim_data_values = spline_sim(exp_time_values)
    sse = np.sum((exp_data_values - sim_data_values) ** 2)
    sse_score = np.exp2(np.log10(sse))
    return [sse_score, ]


def init_score_dict():
    # dictionary of scores with tuples as
    # (score function, [list, of, score, names, returned, by, function])
    score_dict = Dict()
    score_dict["bt"] = (bt_score, ["mid_on", "upper_delay", "max_val"])
    score_dict["bt2"] = (bt_score2, ["mid_on", "delay_85", "delay_95", "max_val"])
    score_dict["bt2_ranged"] = (create_ranged_function(bt_score2), ["mid_on", "delay_85", "delay_95", "max_val"])

    score_dict["peak_height_log"] = (get_peak_height_log_score, ["height_log"])
    score_dict["peak_height"] = (get_peak_height_score, ["height"])
    score_dict["peak_height_ranged"] = (create_ranged_function(get_peak_height_score), ["height"])

    score_dict["spline_time"] = (get_time_score, ["time"])
    score_dict["spline_time_ranged"] = (create_ranged_function(get_time_score), ["time"])

    score_dict["skew"] = (get_skewness_spline_score, ["skew"])
    score_dict["skew_ranged"] = (create_ranged_function(get_skewness_spline_score), ["skew"])

    score_dict["skew_10percent"] = (get_skewness_10percent_score, ["skew"])
    score_dict["skew_10percent_ranged"] = (create_ranged_function(get_skewness_10percent_score), ["skew"])

    score_dict["skew_nospline"] = (get_skewness_score, ["skew_nospline"])
    score_dict["skew_nospline_ranged"] = (create_ranged_function(get_skewness_score), ["skew_nospline"])

    score_dict["spline_shape"] = (get_shape_score, ["shape"])
    score_dict["spline_shape_ranged"] = (create_ranged_function(get_shape_score), ["shape"])

    score_dict["spline"] = (get_time_and_shape_score, ["time", "shape"])
    score_dict["pre_grad_elu"] = (pre_gradient_elution, ["delta"])
    score_dict["SSE"] = (get_sse_score, ["SSE"])
    score_dict["SSE_ranged"] = (create_ranged_function(get_sse_score), ["SSE"])
    score_dict["SSE_nonpen"] = (create_ranged_function(get_sse_nonpen_score), ["SSE"])
    score_dict["SSE_nonpen_ranged"] = (create_ranged_function(get_sse_nonpen_score), ["SSE"])
    return score_dict

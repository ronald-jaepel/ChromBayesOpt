import shutil
import traceback
from datetime import datetime, timedelta
import json
from shutil import rmtree

import matplotlib as mpl
import matplotlib.pyplot as plt
from logging import getLogger

from statsmodels.stats.weightstats import DescrStatsW

from smt.sampling_methods import LHS
import numpy as np
import pandas as pd
import os
from os.path import join
from addict import Dict
from joblib import Parallel, delayed
from search_package.bayes_opt_custom import UtilityFunction, BayesianOptimizationMulti
from search_package.cadet_interface import load_template_sim, load_data_target, \
    preapply_exp_info, plot_sim

"""init data structures"""


def turn_parameter_list_to_dict(json_dict):
    if type(json_dict.parameters) == list:
        for parameter in json_dict["parameters"]:
            if "name" not in parameter:
                if parameter["component"] == -1:
                    parameter["name"] = parameter["location"].split("/")[-1]
                else:
                    parameter["name"] = parameter["location"].split("/")[-1] + "_c" + str(
                        parameter["component"])
            if parameter["lim_min"] > parameter["min"]:
                raise ValueError(f"Parameter {parameter['name']} lim_min was above the min.")
                # parameter["lim_min"] = parameter["min"]
            if parameter["lim_max"] < parameter["max"]:
                raise ValueError(f"Parameter {parameter['name']} lim_max was below the max.")
                # parameter["lim_max"] = parameter["max"]
        if not len({param["name"] for param in json_dict["parameters"]}) == len(
                json_dict["parameters"]):
            raise ValueError(f"names for parameters are not unique in {json_dict.parameters}")
        json_dict["parameters"] = Dict({param["name"]: param for param in json_dict["parameters"]})
        for param in json_dict["parameters"].values():
            param.pop("name")
    return json_dict


def divide_weights_by_scorenumber(json_dict):
    for experiment_name, experiment in json_dict.experiments.items():
        combined_number_of_scores = 0
        for feature in experiment.features.values():
            combined_number_of_scores += len(feature.weights)
        for feature in experiment.features.values():
            feature.weights = [weight / combined_number_of_scores for weight in feature.weights]
    return json_dict


def turn_experiment_list_to_dict(json_dict):
    if type(json_dict.experiments) == list:
        if not len({exp["name"] for exp in json_dict.experiments}) == len(json_dict.experiments):
            raise ValueError(f"names for experiments are not unique in {json_dict.experiments}")

        json_dict.experiments = Dict({exp["name"]: exp for exp in json_dict.experiments})
        for exp in json_dict.experiments:
            exp.pop("name")

    for exp_name, exp in json_dict.experiments.items():

        if type(exp.features) == list:
            if not len({feature["name"] for feature in exp.features}) == len(exp.features):
                raise ValueError(
                    f"names for features are not unique in {exp_name} : {exp.features}")
            json_dict.experiments[exp_name].features = Dict(
                {f'{exp_name}_{feature["name"]}': feature for feature in exp.features})
            for f_name, feature in exp.features.items():
                feature.pop("name")

        for feature in exp.features.values():
            if not "isotherm" in feature:
                feature.isotherm = exp.isotherm
            if not "CSV" in feature:
                feature.CSV = exp.CSV
            if "time_selection" in exp and "time_selection" not in feature:
                feature.time_selection = exp.time_selection
    return json_dict


def bounds_dict_from_json(json_dict):
    bounds_dict = Dict()
    for param_name, param_dict in json_dict.parameters.items():
        bounds_dict[param_name] = (param_dict.min, param_dict.max)
    return bounds_dict


def init_directories(json_dict):
    os.makedirs(join(json_dict.baseDir, json_dict.resultsDir), exist_ok=True)
    os.makedirs(json_dict.fig_base_path, exist_ok=True)
    os.makedirs(json_dict.fig_base_path + "_sims", exist_ok=True)
    os.makedirs(json_dict.sim_tmp_base_path, exist_ok=True)
    if "sim_source_base_path" not in json_dict:
        raise ValueError(f"sim_source_base_path not specified on {[json_dict.baseDir, json_dict.resultsDir]}")
    os.makedirs(json_dict.sim_source_base_path, exist_ok=True)
    with open(join(json_dict.baseDir, json_dict.resultsDir, "meta.json"), 'w') as outfile:
        json.dump(json_dict, outfile, indent=4)
    with open(join(
            json_dict.baseDir,
            json_dict.resultsDir,
            f"meta_archive_{datetime.now().strftime('%Y-%m-%d %H-%M')}.json"), 'w') as outfile:
        json.dump(json_dict, outfile, indent=4)
    for exp_name, exp_dict in sorted(json_dict.experiments.items()):
        h5_source_path = join(json_dict.baseDir, exp_dict["HDF5"])
        h5_target_path = join(json_dict.sim_source_base_path, exp_dict["HDF5"])
        shutil.copyfile(h5_source_path, h5_target_path)


def find_threshold_score(res_space, score_threshold_count, score_threshold_factor):
    res_space_sorted = res_space.sort_values("st", ascending=True)
    best_params = res_space_sorted.iloc[0, :]
    # """ shrink boundaries """
    # message += "shrinking\r\n"
    if score_threshold_count is not None and score_threshold_factor is None:
        try:
            nth_best_params = res_space_sorted.iloc[score_threshold_count, :]
        except IndexError:
            nth_best_params = res_space_sorted.iloc[-1, :]
        threshold_score = nth_best_params.st
    elif score_threshold_count is None and score_threshold_factor is not None:
        threshold_score = best_params.st * score_threshold_factor
    elif score_threshold_count is not None and score_threshold_factor is not None:
        try:
            nth_best_params = res_space_sorted.iloc[score_threshold_count, :]
        except IndexError:
            nth_best_params = res_space_sorted.iloc[-1, :]
        threshold_score = min(nth_best_params.st, best_params.st * score_threshold_factor)
    else:
        raise ValueError
    return threshold_score


def check_boundaries(json_dict, res_space, score_threshold_factor=None, score_threshold_count=None):
    res_space_sorted = res_space.sort_values("st", ascending=True)
    best_params = res_space_sorted.iloc[0, :]
    message = ["0_readjusting parameters\r\n"]

    threshold_score = find_threshold_score(res_space, score_threshold_count, score_threshold_factor)

    limited_df = res_space[res_space.st < threshold_score]
    if limited_df.shape[0] < 2:
        limited_df = res_space_sorted.iloc[:2, :]
    for param, param_dict in json_dict.parameters.items():
        if param_dict.experiments == ["just_set"]:
            continue
        message += [f"{param.ljust(20, ' ')} _was "
                    f"{round(param_dict.min, 3), round(param_dict.max, 3)} \r\n"]
        new_max = limited_df.max()[param]
        new_min = limited_df.min()[param]
        # if new_max == new_min:
        #     new_max += abs(new_max/10)
        #     new_min -= abs(new_min/10)
        if new_min >= param_dict.lim_min:
            # message += f"shifting min from {round(param_dict.min, 3)} " \
            #            f"to {round(new_min, 3)}\r\n"
            param_dict.min = new_min
        if new_max <= param_dict.lim_max:
            # message += f"shifting max from {round(param_dict.max, 3)} " \
            #            f"to {round(new_max, 3)}\r\n"
            param_dict.max = new_max

    # """ grow boundaries """
    for param, param_dict in json_dict.parameters.items():
        if param_dict.experiments == ["just_set"]:
            continue
        param_span = param_dict.max - param_dict.min
        edge_grow_size = 3
        edge_detection_size = 0.1
        default_extension_size = 0.2

        lower_edge_detection_boundary = param_dict.min + param_span * edge_detection_size
        upper_edge_detection_boundary = param_dict.max - param_span * edge_detection_size

        if best_params[param] < lower_edge_detection_boundary:
            # if param close (20% of param_span) to lower edge, grow lower edge by 300%
            param_dict.min = max(param_dict.min - param_span * edge_grow_size, param_dict.lim_min)
        elif best_params[param] > upper_edge_detection_boundary:
            # if param close (20% of param_span) to upper edge, grow upper edge by 300%
            param_dict.max = min(param_dict.max + param_span * edge_grow_size, param_dict.lim_max)
        else:
            # grow both edges for safety
            param_dict.min = max(param_dict.min - param_span * default_extension_size, param_dict.lim_min)
            param_dict.max = min(param_dict.max + param_span * default_extension_size, param_dict.lim_max)

        # prevent shrinkage to tiny percentage of space. Maintain at least 1% coverage of space
        param_max_span = param_dict.lim_max - param_dict.lim_min
        smallest_desired_span = param_max_span * 0.01
        if param_span < smallest_desired_span:
            span_delta = smallest_desired_span - param_span
            param_dict.min = max(param_dict.min - span_delta / 2, param_dict.lim_min)
            param_dict.max = min(param_dict.max + span_delta / 2, param_dict.lim_max)

        message += [f"{param.ljust(20, ' ')} is   "
                    f"{round(param_dict.min, 3), round(param_dict.max, 3)} \r\n"]

    message.sort()
    message = "".join(message)
    print(message)
    logger = getLogger(json_dict.resultsDir)
    logger.info(message)
    return json_dict


def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def collect_hof(search_id="", root_dir="/home/scan/git/matching/",
                threshold=0.99,
                max_count=None,
                split_run_info=False,
                write_corr_to_excel=False,
                compute_errors=True,
                return_best_per_exp=False,
                write_first_past=False,
                write_chrono=False,
                results_file_name="results.csv"):
    try:
        np.seterr(all='print')
        hof_subdir = "0_hofs"
        hof_path = os.path.join(root_dir, hof_subdir)
        if not os.path.exists(hof_path):
            os.makedirs(hof_path, exist_ok=True)
        hof_filename = os.path.join(hof_path,
                                    f"hof {datetime.now().strftime('%Y-%m-%d %H-%M')} {search_id}")
        hof_new = []
        hof_first_threshold = []
        score_over_time = []
        if write_corr_to_excel:
            writer_corrs = pd.ExcelWriter(hof_filename + "_cross_corrs.xlsx")
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if not (file == results_file_name and search_id in root):
                    continue
                try:
                    with open(join(root, file), "r") as file_handle:
                        if ";" in file_handle.readline():
                            sep = ";"
                        else:
                            sep = ","

                    res_space = pd.read_csv(join(root, file), index_col=0, sep=";").dropna()
                    res_space.loc[:, "index"] = res_space.index.values
                    if max_count is not None:
                        res_space = res_space.iloc[:max_count + 1, :]
                    if "doe" in search_id:
                        res_name = os.path.join(*splitall(root)[-3:])
                    else:
                        res_name = os.path.join(*splitall(root)[-2:])

                    # get score over time
                    st = res_space.st
                    st = st.rename(res_name)
                    score_over_time.append(st)

                    if write_first_past:
                        first_past_idx = (res_space.st.values < threshold).argmax()
                        first_past = res_space.iloc[first_past_idx, :]
                        first_past = first_past.rename(res_name)
                        res_space.sort_values("st", inplace=True, ascending=True)
                        best = res_space.iloc[0, :]
                        if first_past.st < threshold:
                            first_past = best.rename(res_name)
                        hof_first_threshold.append(first_past)
                    else:
                        res_space.sort_values("st", inplace=True, ascending=True)
                        best = res_space.iloc[0, :]
                    if res_space.shape[0] == 0:
                        continue
                    if compute_errors or write_corr_to_excel:
                        res_space.loc[:, "weights"] = 1 - (res_space.st - res_space.iloc[0, :].st)
                        res_space.loc[res_space["weights"] < 0, "weights"] = 0
                    if write_corr_to_excel:
                        corr = res_space.loc[res_space["weights"] > 0, :].corr()
                        corr_sma = pd.Series(corr.loc["SMA_KA_c1", "SMA_NU_c1"],
                                             index=["crosscor"],
                                             name="crosscorr_ka_nu")
                        corr_sma_keq = pd.Series(corr.loc["SMA_KEQ_c1", "SMA_NU_c1"],
                                                 index=["crosscor"],
                                                 name="crosscorr_keq_nu")
                        res_name_corr = os.path.join(*splitall(root)[-2:-1])
                        corr.to_excel(writer_corrs, sheet_name=res_name_corr[:31])

                    if compute_errors:
                        w1 = DescrStatsW(res_space.iloc[:, :-2].to_numpy(),
                                         weights=res_space["weights"].to_numpy())
                        mean_series = pd.Series(np.append(w1.mean, [1, 0]),
                                                index=[col + "_mean" for col in res_space.columns],
                                                name="mean")
                        std_series = pd.Series(np.append(w1.std, [1, 0]),
                                               index=[col + "_std" for col in res_space.columns],
                                               name="std")
                        std_mean_series = pd.Series(np.append(w1.std_mean, [1, 0]),
                                                    index=[col + "_std_mean" for col
                                                           in res_space.columns],
                                                    name="std_mean")
                        res_space.to_csv(join(root, file[:-4] + "_stds.csv"), sep=";")

                    if compute_errors and write_corr_to_excel:
                        full = pd.concat(
                            [best, std_series,
                             corr_sma, corr_sma_keq]).sort_index()  # mean_series, std_mean_series,
                    elif compute_errors:
                        full = pd.concat(
                            [best, std_series]).sort_index()  # mean_series, std_mean_series,
                    else:
                        full = best.sort_index()
                    full = full.rename(res_name)
                    hof_new.append(full)
                except Exception as e:
                    # if res_space.weights.sum() != 1:
                    print("\r\n\r\n")
                    print(f"****** HOF_ERROR in file {root} ******")
                    traceback.print_exc()
                    print("\r\n\r\n")
                    continue
        if write_corr_to_excel:
            writer_corrs.close()

        if write_chrono:
            writer = pd.ExcelWriter(hof_filename + "_chrono.xlsx")

            score_over_time_df = pd.concat(score_over_time, axis=1)
            min_score_df = score_over_time_df.copy()
            for i in range(1, score_over_time_df.shape[0]):
                min_score_df.iloc[i, :] = min_score_df.iloc[i - 1:i + 1, :].min(axis=0,
                                                                                skipna=True)
            log_min_score_df = -np.log10(min_score_df)

            score_over_time_df.to_excel(writer, sheet_name="score_over_time")
            min_score_df.to_excel(writer, sheet_name="min_score")
            log_min_score_df.to_excel(writer, sheet_name="log_min_score")
            writer.close()

        hof_best_df = pd.concat(hof_new, axis=1, sort=True).T
        hof_best_df = hof_best_df.sort_values("st", ascending=True)
        split_index = pd.Series([os.path.split(x) for x in hof_best_df.index])
        if return_best_per_exp:
            if len(split_index[0]) == 2:
                multi_index = pd.MultiIndex.from_tuples(split_index,
                                                        names=["exp_id", "search"])
            else:
                multi_index = pd.MultiIndex.from_tuples(split_index,
                                                        names=["parent_dir", "columns", "search"])

            hof_best_df["exp_id"] = multi_index.get_level_values(0)
            # groupy by exp_id, get max of st-score of each exp_id group, compare for each row if that
            # row is the max, true for max, false for non-max
            is_row_max_idx = hof_best_df.groupby(['exp_id'])['st'].transform(max) == hof_best_df['st']
            # trim to best of experiment
            hof_best_df = hof_best_df[is_row_max_idx]
            hof_best_df = hof_best_df.drop("exp_id", axis=1)
            split_index = split_index[is_row_max_idx.values]
            multi_index = multi_index[is_row_max_idx.values]
        if split_run_info:
            split_index = split_index.apply(lambda x: [x[1]] + x[0].split("_"))
            split_index = split_index.apply(lambda x: x[0:2] + x[2].split(" ") + x[3:])
            # split_index = split_index.apply(lambda x: x if len(x) == 12 else x + ["0.5M"])
            split_index = split_index.apply(lambda x: x + [None] * (12 - len(x)))
            multi_index = pd.MultiIndex.from_tuples(split_index, names=[
                "search", "date", "column", "resin", "res_batch", "protein", "uniprot",
                "prot_batch", "pH", "extra_salt", "buffer", "final_salt"])
        else:
            multi_index = pd.MultiIndex.from_tuples(split_index,
                                                    names=["parent_dir", "columns", "search"])
        hof_best_df = hof_best_df.set_index(multi_index)
        cols = hof_best_df.columns.tolist()
        cols.remove("st")
        cols.remove("index")
        if "index_st" in cols:
            cols.remove("index_st")
        cols = ["st", "index"] + cols
        hof_best_df = hof_best_df.loc[:, cols]
        if max_count is not None:
            best_xlsx_filename = hof_filename + f"_best_{max_count}.xlsx"
        else:
            best_xlsx_filename = hof_filename + "_best.xlsx"
        writer = pd.ExcelWriter(best_xlsx_filename)
        hof_best_df.to_excel(writer, sheet_name="1", merge_cells=False)
        writer.close()

        if write_first_past:
            hof_first_threshold_df = pd.concat(hof_first_threshold, axis=1, sort=True).T
            try:
                split_index = pd.Series([os.path.split(x) for x in hof_best_df.index])
                multi_index = pd.MultiIndex.from_tuples(split_index,
                                                        names=["parent_dir", "columns", "search"])
                hof_first_threshold_df = hof_first_threshold_df.set_index(multi_index)
            except TypeError:
                pass
            writer = pd.ExcelWriter(hof_filename + "_first_past.xlsx")
            hof_first_threshold_df.to_excel(writer, merge_cells=False)
            writer.close()
        return best_xlsx_filename
    except Exception as e:
        print("\r\n\r\n")
        print(f"****** HOF_ERROR in concatenation")
        traceback.print_exc()
        print("\r\n\r\n")
        return


def init_target_functions_and_sims(json_dict, score_dict, add_target_into_hdf5=False):
    sim_list = []
    target_function_list = []

    for exp_name, exp_dict in sorted(json_dict.experiments.items()):
        if "sim_source_base_path" not in json_dict:
            raise ValueError(f"sim_source_base_path not specified on {[json_dict.baseDir, json_dict.resultsDir]}")
        template_sim = load_template_sim(exp_dict["HDF5"], json_dict.sim_source_base_path,
                                         json_dict.CADETPath)

        # check if individual features of experiments
        # have individual CSV data targets
        if all("CSV" in feature for f_name, feature in exp_dict["features"].items()) and \
                len(exp_dict["features"]) != 0:
            data_targets = \
                {feature["CSV"]: load_data_target(feature["CSV"], json_dict.baseDir)
                 for f_name, feature in exp_dict["features"].items()}
        else:
            data_targets = \
                {exp_dict["CSV"]: load_data_target(exp_dict["CSV"], json_dict.baseDir)}
        if add_target_into_hdf5:
            data_target = data_targets[list(data_targets.keys())[0]]
            template_sim.root.meta["data_target"] = np.array(data_target.signal)
        sim_list.append(template_sim)

        experiment_target_function = preapply_exp_info(exp_name, template_sim, data_targets,
                                                       json_dict, score_dict)
        target_function_list.append(experiment_target_function)
    return sim_list, target_function_list


def compile_feature_score_names_list(json_dict, score_dict):
    for exp_name, exp_dict in json_dict.experiments.items():
        feature_score_name_list = []
        for featurename, feature in sorted(exp_dict.features.items()):
            if feature.type not in score_dict:
                raise ValueError(f"No score of type {feature.type} in score dictionary.")
            score_name_list = score_dict[feature.type][1]
            score_name_list = [f"{exp_name}-{featurename}-{score_name}" for score_name in score_name_list]
            feature_score_name_list.extend(score_name_list)
        exp_dict["feature_score_names"] = feature_score_name_list
    return json_dict


def calculate_mask(json_dict):
    """
    Mask is an array of shape (parameters, experiment_features) which assigns
    each parameter - experiment_feature combination a boolean
    Both whole experiments can be added by including the experiment
    name in the parameter.experiments list or only individual features
    by adding the feature name
    :param json_dict:
    :return:
    """
    mask_list = []
    for param_name, param in json_dict.parameters.items():
        if param.features == {}:
            print(f"No features set for parameter {param_name}, using param.experiments instead")
            param.features = param.experiments
    for exp_name, exp_dict in sorted(json_dict.experiments.items()):
        for feature_score_name in sorted(exp_dict.feature_score_names):
            mask_list.append([exp_name in param.features
                              or any(score in feature_score_name for score in param.features)
                              for param_name, param in sorted(json_dict.parameters.items())])

    mask = np.array(mask_list).T
    mask_df = pd.DataFrame(mask,
                           columns=[feature_score_name
                                    for exp_name, exp_dict in sorted(json_dict.experiments.items())
                                    for feature_score_name in sorted(exp_dict.feature_score_names)],
                           index=list(json_dict.parameters.keys()))
    return mask


def init_in_bounds(json_dict, bounds, bounds_dict, target_function, optimizer, previous_k=1):
    """
    Initialize the search space with a latin hypercube sampling
    """
    remaining_x = optimizer._gp.filtered_x_shape[0]
    if remaining_x is None:
        n_init = json_dict["warmup_points"]
    else:
        n_init = max(4, int(json_dict["warmup_points"] - remaining_x))
    if n_init < 3:
        n_init = 3
    samples = LHS(xlimits=bounds, criterion="ese", random_state=json_dict["seed"])(n_init)

    samples = np.unique(samples, axis=0)

    que = [{key: samples[k, idx]
            for idx, key in enumerate(sorted(bounds_dict))}
           for k in range(samples.shape[0])]

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    que_chunks_size = 20
    que_chunks = chunks(que, que_chunks_size)

    if "target" not in optimizer.min:
        previous_best = 100
    else:
        previous_best = optimizer.min["target"]

    k = previous_k + 1
    for sub_que in que_chunks:
        for point in sub_que:
            target = target_function(uuid=k, **point)
            optimizer.register(params=point, target=target)
            if json_dict.create_plots and target[-1] < previous_best:
                previous_best = target[-1]
                plot_sim(str(k), json_dict)
            k += 1
        rmtree(json_dict.sim_tmp_base_path)
        os.makedirs(json_dict.sim_tmp_base_path)

    return optimizer


"""helper functions"""


def load_previous_results(optimizer, file_path, bounds_dict, index_limit=None,
                          return_res_space=False):
    sep = ","
    with open(file_path, "r") as file_handle:
        if ";" in file_handle.readline():
            sep = ";"

    res_space = pd.read_csv(file_path, index_col=0, sep=sep)
    res_space.dropna(axis=0, how="all", inplace=True)
    res_space.dropna(axis=1, how="all", inplace=True)
    res_space.replace("#VALUE!", np.nan, inplace=True)
    res_space = res_space.astype(float)
    # res_space[res_space.loc[:, "pulse_60cv-pulse_60cv_height_1-height_log"] < -1.5] = -1.5
    # res_space[res_space.loc[:, "pulse_60cv-pulse_60cv_height_2-height_log"] < -1.5] = -1.5
    if index_limit is not None:
        res_space = res_space.iloc[:index_limit, :]

    for i, row in res_space.iterrows():
        point = row.iloc[0:len(bounds_dict)].to_dict()
        target = row.iloc[len(bounds_dict):].values
        optimizer.register(params=point, target=target)
    if return_res_space:
        return optimizer, res_space
    else:
        return optimizer


def collect_res_space(optimizer, bounds_dict, gp_name_list, results_csv_file):
    res_params = optimizer.space.params
    res_y = optimizer.space.raw_target

    _res_space = pd.DataFrame(np.concatenate((res_params, res_y), axis=1),
                              columns=list(sorted(bounds_dict)) + gp_name_list + ["st", ])
    # _res_space_abs = pd.DataFrame(np.concatenate((res_params, res_y), axis=1),
    #                               columns=list(sorted(bounds_dict)) + gp_name_list + ["st", ])
    if _res_space.shape[0] > 0:
        _res_space.to_csv(results_csv_file, sep=";", na_rep="nan")
        _res_space.to_csv(results_csv_file.replace(".csv", "") +
                          f"{datetime.now().strftime('%Y-%m-%d %H')}.csv", sep=";", na_rep="nan")

    _res_space.sort_values("st", inplace=True, ascending=True)
    return _res_space


def find_squarest_arrangement(n, shape_per_box=(1, 1)):
    """

    :param n: number of boxes
    :param shape_per_box: tuple of (rows, columns)
    :return: rows, columns
    """
    smallest_square = 1e10
    smallest_row = None
    smallest_cols = None
    for n_box_rows in range(1, n):
        box_columns = np.ceil(n / n_box_rows)
        rows = n_box_rows * shape_per_box[0]
        columns = box_columns * shape_per_box[1]
        rectangle = rows + columns
        if rectangle < smallest_square:
            smallest_square = rectangle
            smallest_row = n_box_rows
            smallest_cols = box_columns
    return int(smallest_row), int(smallest_cols)


def surface_plot(slice_point, optimizer, utility, bounds_dict, gps, gp_name_list,
                 close_figures, json_dict, itera):
    def get_index_from_bounds_dict_and_col_within_box(c_within_box, bounds_array):
        if sorted(bounds_array) == ['COL_DISPERSION', 'COL_POROSITY', 'FILM_DIFFUSION',
                                    'PAR_POROSITY', 'SMA_KA_c1', 'SMA_NU_c1']:
            if c_within_box == 0:
                dim_i_0 = 1
                dim_i_1 = 3
            elif c_within_box == 1:
                dim_i_0 = 0
                dim_i_1 = 2
            else:
                dim_i_0 = 4
                dim_i_1 = 5
        elif sorted(bounds_array) == ['COL_DISPERSION', 'COL_POROSITY', 'FILM_DIFFUSION',
                                      'SMA_KA_c1', 'SMA_NU_c1', 'TOT_POROSITY']:
            if c_within_box == 0:
                dim_i_0 = 1
                dim_i_1 = 5
            elif c_within_box == 1:
                dim_i_0 = 0
                dim_i_1 = 2
            else:
                dim_i_0 = 3
                dim_i_1 = 4
        elif sorted(bounds_array) == ['COL_DISPERSION', 'COL_POROSITY', 'FILM_DIFFUSION',
                                      'PAR_POROSITY']:
            if c_within_box == 0:
                dim_i_0 = 1
                dim_i_1 = 3
            else:
                dim_i_0 = 0
                dim_i_1 = 2

        else:
            if c_within_box * 2 + 1 == len(bounds_array):
                dim_i_0 = c_within_box * 2 - 1
                dim_i_1 = c_within_box * 2 + 0
            else:
                dim_i_0 = c_within_box * 2 + 0
                dim_i_1 = c_within_box * 2 + 1
        return dim_i_0, dim_i_1

    if slice_point is None:
        slice_point = optimizer.suggest(utility, n_iter=2, n_warmup=100)
    slice_point = {param: slice_point[param] for param in sorted(bounds_dict)}

    fitted_gp_list = gps.gp_list
    y_max = max(optimizer.space.raw_target[:, -1])

    n_boxes = len(fitted_gp_list) + 2
    shape_graphs_per_box = (2, int(np.ceil(len(bounds_dict) / 2)))
    box_rows, box_cols = find_squarest_arrangement(n_boxes, shape_graphs_per_box)
    n_rows, n_cols = box_rows * shape_graphs_per_box[0], box_cols * shape_graphs_per_box[1]
    box_i = 0

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3, n_rows * 2),
                             sharex="col", sharey="col")
    fig2, axes2 = plt.subplots(n_rows, n_cols,
                               figsize=(n_cols * 3, n_rows * 2),
                               sharex="col", sharey="col")

    # PLot individual gps + std
    for box_i in range(len(fitted_gp_list)):
        for col_within_box in range(shape_graphs_per_box[1]):
            dim_index_0, dim_index_1 = get_index_from_bounds_dict_and_col_within_box(
                col_within_box,
                bounds_dict)
            dim0 = sorted(bounds_dict)[dim_index_0]
            dim1 = sorted(bounds_dict)[dim_index_1]
            dims_fixed = set(bounds_dict) - {dim0, dim1}

            x0_g, x1_g = np.meshgrid(np.linspace(*bounds_dict[dim0], num=20),
                                     np.linspace(*bounds_dict[dim1], num=20))
            x0_f = x0_g.flatten().reshape(-1, 1)
            x1_f = x1_g.flatten().reshape(-1, 1)
            array_dict = {dim: np.ones_like(x0_f) * slice_point[dim] for dim in dims_fixed}
            array_dict[dim0] = x0_f
            array_dict[dim1] = x1_f
            array_list = [array_dict[key] for key in sorted(array_dict.keys())]
            x = np.concatenate(array_list, axis=1)

            y_mean, y_std = gps.predict_individual_gp(x, box_i, return_std=True)
            vmax = max(np.abs(y_mean.min()), y_mean.max())
            # To increase the overall saturation of plots
            vmax = vmax / 1.2

            box_r = box_i // box_cols
            box_c = box_i % box_cols
            r = box_r * 2
            c = box_c * shape_graphs_per_box[1] + col_within_box

            contr = axes[r, c].contourf(x0_g, x1_g, y_mean.reshape(x0_g.shape), 40,
                                        cmap="seismic",
                                        vmin=-vmax, vmax=vmax)
            fig.colorbar(contr, ax=axes[r, c],
                         ticks=sorted([y_mean.min(), 0, y_mean.mean(), y_mean.max()]))
            axes[r, c].set_xlim(auto=False)
            axes[r, c].set_ylim(auto=False)
            contr = axes[r + 1, c].contourf(x0_g, x1_g, y_std.reshape(x0_g.shape), 40,
                                            cmap="binary", vmin=0, vmax=vmax*0.5)
            fig.colorbar(contr, ax=axes[r + 1, c],
                         ticks=sorted([0, y_std.min(), y_std.mean(), y_std.max(), vmax*0.5]))
            axes[r + 1, c].set_xlim(auto=False)
            axes[r + 1, c].set_ylim(auto=False)
            axes[r, c].set_xlabel(dim0)
            axes[r, c].set_ylabel(dim1)
            axes[r + 1, c].set_xlabel(dim0)
            axes[r + 1, c].set_ylabel(dim1)
            axes[r, c].set_title(gp_name_list[box_i][-25:])

            contr = axes2[r, c].contourf(x0_g, x1_g, y_mean.reshape(x0_g.shape), 40,
                                         cmap="seismic",
                                         vmin=-1, vmax=1)
            fig.colorbar(contr, ax=axes2[r, c],
                         ticks=sorted([y_mean.min(), 0, y_mean.mean(), y_mean.max()]))
            axes2[r, c].set_xlim(auto=False)
            axes2[r, c].set_ylim(auto=False)
            contr = axes2[r + 1, c].contourf(x0_g, x1_g, y_std.reshape(x0_g.shape), 40,
                                             cmap="binary", vmin=0, vmax=0.5)
            fig.colorbar(contr, ax=axes2[r + 1, c],
                         ticks=sorted([y_std.min(), y_std.mean(), y_std.max(), vmax]))
            axes2[r + 1, c].set_xlim(auto=False)
            axes2[r + 1, c].set_ylim(auto=False)
            axes2[r, c].set_xlabel(dim0)
            axes2[r, c].set_ylabel(dim1)
            axes2[r + 1, c].set_xlabel(dim0)
            axes2[r + 1, c].set_ylabel(dim1)
            axes2[r, c].set_title(gp_name_list[box_i][-25:])

    # plot average_of_scores + std
    x_list = []
    y_mean_list = []
    y_std_list = []
    box_i += 1
    for col_within_box in range(shape_graphs_per_box[1]):
        dim_index_0, dim_index_1 = get_index_from_bounds_dict_and_col_within_box(col_within_box,
                                                                                 bounds_dict)
        dim0 = sorted(bounds_dict)[dim_index_0]
        dim1 = sorted(bounds_dict)[dim_index_1]
        dims_fixed = set(bounds_dict) - {dim0, dim1}

        x0_g, x1_g = np.meshgrid(np.linspace(*bounds_dict[dim0], num=20),
                                 np.linspace(*bounds_dict[dim1], num=20))
        x0_f = x0_g.flatten().reshape(-1, 1)
        x1_f = x1_g.flatten().reshape(-1, 1)
        array_dict = {dim: np.ones_like(x0_f) * slice_point[dim] for dim in dims_fixed}
        array_dict[dim0] = x0_f
        array_dict[dim1] = x1_f
        array_list = [array_dict[key] for key in sorted(array_dict.keys())]
        x = np.concatenate(array_list, axis=1)

        y_mean, y_std = gps.predict(x, return_std=True)
        x_list.append(x)
        y_mean_list.append(y_mean)
        y_std_list.append(y_std)

        box_r = box_i // box_cols
        box_c = box_i % box_cols
        r = box_r * 2
        c = box_c * shape_graphs_per_box[1] + col_within_box

        contr = axes[r, c].contourf(x0_g, x1_g, y_mean.reshape(x0_g.shape), 40)
        cb = fig.colorbar(contr, ax=axes[r, c],
                          ticks=sorted([y_mean.min(), 0, y_mean.mean(), y_mean.max()]))
        axes[r, c].set_xlim(auto=False)
        axes[r, c].set_ylim(auto=False)
        contr = axes[r + 1, c].contourf(x0_g, x1_g, y_std.reshape(x0_g.shape), 40,
                                        cmap="binary",
                                        vmin=0, vmax=1)
        fig.colorbar(contr, ax=axes[r + 1, c],
                     ticks=sorted([y_std.min(), y_std.mean(), y_std.max(), 1]))
        axes[r + 1, c].set_xlim(auto=False)
        axes[r + 1, c].set_ylim(auto=False)
        axes[r, c].set_xlabel(dim0)
        axes[r, c].set_ylabel(dim1)
        axes[r + 1, c].set_xlabel(dim0)
        axes[r + 1, c].set_ylabel(dim1)
        axes[r, c].set_title("Average of scores")

        axes2[r, c].set_xlim(auto=False)
        axes2[r, c].set_ylim(auto=False)
        contr = axes2[r + 1, c].contourf(x0_g, x1_g, y_std.reshape(x0_g.shape), 40,
                                         cmap="binary",
                                         vmin=0, vmax=1)
        fig.colorbar(contr, ax=axes2[r + 1, c],
                     ticks=sorted([y_std.min(), y_std.mean(), y_std.max(), 1]))
        axes2[r + 1, c].set_xlim(auto=False)
        axes2[r + 1, c].set_ylim(auto=False)
        axes2[r, c].set_xlabel(dim0)
        axes2[r, c].set_ylabel(dim1)
        axes2[r + 1, c].set_xlabel(dim0)
        axes2[r + 1, c].set_ylabel(dim1)
        axes2[r, c].set_title("Average of scores")

    # plot utility
    box_i += 1
    for col_within_box in range(shape_graphs_per_box[1]):
        dim_index_0, dim_index_1 = get_index_from_bounds_dict_and_col_within_box(col_within_box,
                                                                                 bounds_dict)
        dim0 = sorted(bounds_dict)[dim_index_0]
        dim1 = sorted(bounds_dict)[dim_index_1]
        dims_fixed = set(bounds_dict) - {dim0, dim1}

        x0_g, x1_g = np.meshgrid(np.linspace(*bounds_dict[dim0], num=10),
                                 np.linspace(*bounds_dict[dim1], num=10))
        x0_f = x0_g.flatten().reshape(-1, 1)
        x1_f = x1_g.flatten().reshape(-1, 1)
        array_dict = {dim: np.ones_like(x0_f) * slice_point[dim] for dim in dims_fixed}
        array_dict[dim0] = x0_f
        array_dict[dim1] = x1_f
        array_list = [array_dict[key] for key in sorted(array_dict.keys())]
        x = np.concatenate(array_list, axis=1)

        util_space = utility.utility(x, gps, y_max)
        utility_ucb = UtilityFunction(kind="ucb", kappa=1, xi=1e-2, n_samples=10000)
        util_space_ucb = utility_ucb.utility(x, gps, y_max)

        box_r = box_i // box_cols
        box_c = box_i % box_cols
        r = box_r * 2
        c = box_c * shape_graphs_per_box[1] + col_within_box

        contr = axes[r, c].contourf(x0_g, x1_g, util_space.reshape(x0_g.shape), 40, cmap="viridis_r")
        fig.colorbar(contr, ax=axes[r, c],
                     ticks=[y_mean.min(), y_mean.mean(), y_mean.max()])
        axes[r, c].set_xlim(auto=False)
        axes[r, c].set_ylim(auto=False)

        contr = axes[r + 1, c].contourf(x0_g, x1_g, util_space_ucb.reshape(x0_g.shape), 40, cmap="viridis_r")
        fig.colorbar(contr, ax=axes[r + 1, c],
                     ticks=[y_mean.min(), y_mean.mean(), y_mean.max()])
        axes[r + 1, c].set_xlim(auto=False)
        axes[r + 1, c].set_ylim(auto=False)
        axes[r, c].set_xlabel(dim0)
        axes[r, c].set_ylabel(dim1)
        axes[r + 1, c].set_xlabel(dim0)
        axes[r + 1, c].set_ylabel(dim1)
        axes[r, c].set_title("Utility UCB k=0")
        axes[r + 1, c].set_title("Utility UCB k=1")

        contr = axes2[r, c].contourf(x0_g, x1_g, util_space.reshape(x0_g.shape), 40, cmap="viridis_r")
        fig.colorbar(contr, ax=axes2[r, c],
                     ticks=[y_mean.min(), y_mean.mean(), y_mean.max()])
        axes2[r, c].set_xlim(auto=False)
        axes2[r, c].set_ylim(auto=False)

        contr = axes2[r + 1, c].contourf(x0_g, x1_g, util_space_ucb.reshape(x0_g.shape), 40, cmap="viridis_r")
        fig.colorbar(contr, ax=axes2[r + 1, c],
                     ticks=[y_mean.min(), y_mean.mean(), y_mean.max()])
        axes2[r + 1, c].set_xlim(auto=False)
        axes2[r + 1, c].set_ylim(auto=False)
        axes2[r, c].set_xlabel(dim0)
        axes2[r, c].set_ylabel(dim1)
        axes2[r + 1, c].set_xlabel(dim0)
        axes2[r + 1, c].set_ylabel(dim1)
        axes2[r, c].set_title("Utility UCB k=0")
        axes2[r + 1, c].set_title("Utility UCB k=1")

    fig.suptitle(str(itera) + str({key: np.round(val, 2) for key, val in slice_point.items()}))
    fig2.suptitle(str(itera) + str({key: np.round(val, 2) for key, val in slice_point.items()}))
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig2.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(join(json_dict["fig_base_path"],
                     f"surface_{datetime.now().strftime('%Y-%m-%d %H-%M')}_{itera}.png"),
                dpi=150)
    fig2.savefig(join(json_dict["fig_base_path"],
                      f"surface_scale_{datetime.now().strftime('%Y-%m-%d %H-%M')}_{itera}.png"),
                 dpi=150)
    if close_figures:
        plt.close(fig)
        plt.close(fig2)
    return x_list, y_mean_list, y_std_list


def scatter_all_dims(bounds_dict, x_full, y_full, y_std_full, res_y, res_params, y_min,
                     close_figures, json_dict, itera):
    n_box_rows, n_box_cols = find_squarest_arrangement(len(bounds_dict), (2, 1))
    fig, axes = plt.subplots(n_box_rows * 2, n_box_cols, figsize=(19, 11))
    axes = np.atleast_2d(axes)

    try:
        zoomed_in_y_max = sorted(res_y)[30]  # min(sorted(res_y)[-40], y_min * 2)
        zoomed_out_y_max = res_y.max()
    except IndexError:
        zoomed_in_y_max = y_min * 3
        zoomed_out_y_max = res_y.max()
    zoomed_out_y_max *= 1.1

    # axes = axes.flatten()
    line_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i in range(len(bounds_dict)):
        dim = sorted(bounds_dict)[i]
        row = (i // n_box_cols) * 2
        column = i % n_box_cols
        for j in [0, 1]:
            ax = axes[row + j, column]
            # ax.errorbar(x_full[:, i], y_full.flatten(), yerr=y_std_full.flatten(), fmt="x", zorder=1)
            # the next line is commented because these should be 0, but are now huge because points
            # that are outside of the dimensional-trimming (check_boundaries) are predicted
            # ax.errorbar(res_params[:, i], y_mean_total, yerr=y_std_total, fmt="x", zorder=22)
            ax.scatter(res_params[:, i], res_y, zorder=20)
            ax.scatter(res_params[:, i], res_y, marker="x", c=line_colors[1], zorder=22)
            ax.axvline(bounds_dict[dim][0], c=line_colors[3])
            ax.axvline(bounds_dict[dim][1], c=line_colors[3])
            if j:
                ax.set_ylim(y_min - (zoomed_in_y_max - y_min) / 10, zoomed_in_y_max)
                x_range = bounds_dict[dim][1] - bounds_dict[dim][0]
                ax.set_xlim(bounds_dict[dim][0] - x_range / 10, bounds_dict[dim][1] + x_range / 10)
            else:
                ax.set_ylim(-0.01, zoomed_out_y_max)
            ax.set_ylabel("score")
            ax.set_xlabel(sorted(bounds_dict)[i])
    plt.tight_layout()
    plt.savefig(join(json_dict["fig_base_path"],
                     f"scatter__all_{datetime.now().strftime('%Y-%m-%d %H-%M')}_{itera}.png"),
                dpi=100)
    if close_figures:
        plt.close(fig)


def plot(itera, optimizer, gps, bounds_dict, gp_name_list, json_dict, slice_point=None,
         close_figures=False):
    # return
    try:
        plt.ioff()
        # mask = gps.mask
        # is_parameter_used = mask.sum(axis=1) > 0
        #
        # bounds_dict = {p: bounds_dict[p] for (p, used) in zip(sorted(bounds_dict),
        #                                                       is_parameter_used) if used}
        mpl.rcParams['axes.labelsize'] = 'small'
        mpl.rcParams['axes.titlesize'] = 'small'
        mpl.rcParams['figure.titlesize'] = 'small'
        mpl.rcParams['xtick.labelsize'] = 'small'
        mpl.rcParams['ytick.labelsize'] = 'small'
        # mpl.rcParams['font.size'] = 'small'
        utility = UtilityFunction(kind="ucb", kappa=0, xi=1e-2, n_samples=10000)

        gp_name_list_local = gp_name_list + ["total score"]
        y_min = min(optimizer.space.raw_target[:, -1])

        # x_list = []
        # y_mean_list = []
        # y_std_list = []

        x_list, y_mean_list, y_std_list = surface_plot(slice_point, optimizer, utility, bounds_dict,
                                                       gps, gp_name_list, close_figures, json_dict,
                                                       itera)

        x_full = np.concatenate(x_list)
        y_full = np.concatenate(y_mean_list)
        y_std_full = np.concatenate(y_std_list)
        res_params = optimizer.space.params
        res_y_full = optimizer.space.full_target_space
        res_y = res_y_full[:, -1]
        res_y[np.isnan(res_y)] = res_y.max()
        sorting_index = res_y.argsort()[::-1]
        res_params_sorted = res_params[sorting_index]
        res_y_sorted = res_y[sorting_index]
        # y_mean_total, y_std_total = gps.predict(res_params, return_std=True)

        """ Plot Scatter over all dimensions at once """
        scatter_all_dims(bounds_dict, x_full, y_full, y_std_full, res_y, res_params, y_min,
                         close_figures, json_dict, itera)

        """ PLOT SCATTER PER TARGET DIM"""
        # shape_graphs_per_box = (2, 1)
        # box_rows, box_cols = find_squarest_arrangement(res_params.shape[1], shape_graphs_per_box)
        # n_rows, n_cols = box_rows * shape_graphs_per_box[0], box_cols * shape_graphs_per_box[1]
        #
        # for j in range(res_y_full.shape[1]):  # iterate target dims / gp names
        #     fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3))
        #     for i in range(res_params.shape[1]):
        #         box_r = i // box_cols
        #         box_c = i % box_cols
        #         r = box_r * shape_graphs_per_box[0]
        #         c = box_c * shape_graphs_per_box[1]
        #
        #         axes[r, c].scatter(res_params[:, i], 1 - np.abs(res_y_full[:, j]))
        #         axes[r, c].set_xlabel(sorted(bounds_dict)[i])
        #         axes[r, c].set_ylabel(f"1 - abs({gp_name_list_local[j]})")
        #
        #         axes[r + 1, c].scatter(res_params[:, i], 1 - np.abs(res_y_full[:, j]))
        #         axes[r + 1, c].set_ylim(0.9, 1.01)
        #         axes[r + 1, c].set_xlabel(sorted(bounds_dict)[i])
        #         axes[r + 1, c].set_ylabel(f"1 - abs({gp_name_list_local[j]})")
        #     fig.tight_layout()
        #     fig.savefig(join(json_dict["fig_base_path"],
        #                      f"scatter_{gp_name_list_local[j]}_"
        #                      f"{datetime.now().strftime('%Y-%m-%d %H-%M')}_{itera}.png"),
        #                 dpi=300)
        #     if close_figures:
        #         plt.close(fig)
        #
        # """ Scatter per score """
        # shape_graphs_per_box = (1, 1)
        # box_rows, box_cols = find_squarest_arrangement(res_y_full.shape[1], shape_graphs_per_box)
        # n_rows, n_cols = box_rows * shape_graphs_per_box[0], box_cols * shape_graphs_per_box[1]
        #
        # for j in range(res_params.shape[1]):  # iterate input dimensions
        #     fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
        #     try:
        #         axes = axes.flatten()
        #     except:
        #         axes = [axes]
        #     for i in range(res_y_full.shape[1]):
        #         axes[i].scatter(res_params[:, j], 1 - np.abs(res_y_full[:, i]))
        #         axes[i].set_xlabel(sorted(bounds_dict)[j])
        #         axes[i].set_ylabel(f"1 - abs({gp_name_list_local[i]})")
        #     fig.tight_layout()
        #     fig.savefig(join(json_dict["fig_base_path"],
        #                      f"scatter_{sorted(bounds_dict)[j]}_"
        #                      f"{datetime.now().strftime('%Y-%m-%d %H-%M')}_{itera}.png"),
        #                 dpi=300)
        #     if close_figures:
        #         plt.close(fig)

        """ scatter pairs """
        fig, axes = plt.subplots(len(bounds_dict), len(bounds_dict),
                                 figsize=(len(bounds_dict) * 3, len(bounds_dict) * 2),
                                 sharex="col")
        for i in range(len(bounds_dict)):
            for j in range(len(bounds_dict)):
                if i == j:
                    axes[i, j].scatter(res_params[:, i], res_y, c=1 / res_y, zorder=20,
                                       edgecolors='black', linewidth=0.5)
                    axes[i, j].set_yscale("log")
                    axes[i, j].set_ylabel("score")
                    axes[i, j].set_xlabel(sorted(bounds_dict)[i])
                else:
                    axes[i, j].scatter(res_params_sorted[:, j], res_params_sorted[:, i], c=1 / res_y_sorted, zorder=1,
                                       edgecolors='black', linewidth=0.5)
                    axes[i, j].set_ylabel(sorted(bounds_dict)[i])
                    axes[i, j].set_xlabel(sorted(bounds_dict)[j])
        plt.tight_layout()
        plt.savefig(join(json_dict["fig_base_path"],
                         f"scatter_pairs_{datetime.now().strftime('%Y-%m-%d %H-%M')}_{itera}.png"),
                    dpi=100)

        for i in range(len(bounds_dict)):
            for j in range(len(bounds_dict)):
                if i == j:
                    axes[i, j].set_yscale("linear")
                    zoomed_in_y_max = sorted(res_y)[int(len(res_y) * 0.5)]
                    y_range = zoomed_in_y_max - y_min
                    y_lower_limit = y_min - y_range * 0.05
                    axes[i, j].set_ylim(y_lower_limit, zoomed_in_y_max + y_range * 0.05)
                if i != j:
                    axes[i, j].set_xlim(bounds_dict[sorted(bounds_dict)[j]])
                    axes[i, j].set_ylim(bounds_dict[sorted(bounds_dict)[i]])
        plt.savefig(join(json_dict["fig_base_path"],
                         f"scatter_pairs_bounds_{datetime.now().strftime('%Y-%m-%d %H-%M')}_{itera}.png"),
                    dpi=100)

        """ plot slices """
        fig, axes = plt.subplots(len(bounds_dict), len(bounds_dict),
                                 figsize=(len(bounds_dict) * 3, len(bounds_dict) * 2),
                                 sharex="col")
        for i in range(len(bounds_dict)):
            for j in range(len(bounds_dict)):
                if i == j:
                    # axes[i, j].set_yscale("log")
                    axes[i, j].scatter(res_params[:, i], res_y, c=1 / res_y, zorder=20)
                    zoomed_in_y_max = sorted(res_y)[int(len(res_y) * 0.5)]
                    # axes[i, j].set_ylim(res_y.min() * 0.9, res_y.max()*1.1)
                    y_range = zoomed_in_y_max - y_min
                    y_lower_limit = y_min - y_range * 0.05
                    axes[i, j].set_ylim(y_lower_limit, zoomed_in_y_max + y_range * 0.05)
                    axes[i, j].set_ylabel("score")
                    axes[i, j].set_xlabel(sorted(bounds_dict)[i])
                    axes[i, j].set_xlim(bounds_dict[sorted(bounds_dict)[j]])
                    continue

                dim0 = sorted(bounds_dict)[j]
                dim1 = sorted(bounds_dict)[i]
                dims_fixed = set(bounds_dict) - {dim0, dim1}

                x0_g, x1_g = np.meshgrid(np.linspace(*bounds_dict[dim0], num=20),
                                         np.linspace(*bounds_dict[dim1], num=20))
                x0_f = x0_g.flatten().reshape(-1, 1)
                x1_f = x1_g.flatten().reshape(-1, 1)
                array_dict = {dim: np.ones_like(x0_f) * slice_point[dim] for dim in dims_fixed}
                array_dict[dim0] = x0_f
                array_dict[dim1] = x1_f
                array_list = [array_dict[key] for key in sorted(array_dict.keys())]
                x = np.concatenate(array_list, axis=1)

                y_mean, y_std = gps.predict(x, return_std=True)

                contr = axes[i, j].contourf(x0_g, x1_g, y_mean.reshape(x0_g.shape), 40, cmap="viridis_r",
                                            vmin=res_y.min(), vmax=res_y.max())
                # fig.colorbar(contr, ax=axes[i, j],
                #              ticks=[y_mean.min(), y_mean.mean(), y_mean.max()])
                axes[i, j].set_xlabel(dim0)
                axes[i, j].set_ylabel(dim1)
        plt.tight_layout()
        plt.savefig(join(json_dict["fig_base_path"],
                         f"slices_{datetime.now().strftime('%Y-%m-%d %H-%M')}_{itera}.png"),
                    dpi=150)
        if close_figures:
            plt.close(fig)

        """ plot histograms """
        # fig, axes = plt.subplots(*find_squarest_arrangement(res_y_full.shape[1], (1, 1)),
        #                          figsize=(len(bounds_dict) * 3, len(bounds_dict) * 2))
        # try:
        #     axes = axes.flatten()
        # except:
        #     axes = [axes]
        # for i in range(res_y_full.shape[1]):
        #     axes[i].hist(res_y_full[:, i], bins=40)
        #     axes[i].set_xlabel(gp_name_list_local[i])
        # fig.tight_layout()
        # fig.savefig(join(json_dict["fig_base_path"],
        #                  f"histograms_{datetime.now().strftime('%Y-%m-%d %H-%M')}_{itera}.png"),
        #             dpi=300)

    except:
        traceback.print_exc()
        logger = getLogger(json_dict.resultsDir)
        logger.error(traceback.format_exc())
    return


def load_time_df(json_dict, call_starttime):
    time_df = pd.read_csv(json_dict.time_file_path, sep=";", index_col=0)
    time_df = time_df.fillna(0)
    time_df.loc[:, "suggest"] = time_df.loc[:, "suggest"].apply(
        lambda x: timedelta(seconds=x))
    time_df.loc[:, "target"] = time_df.loc[:, "target"].apply(
        lambda x: timedelta(seconds=x))
    time_df.loc[:, "fit"] = time_df.loc[:, "fit"].apply(
        lambda x: timedelta(seconds=x))
    time_df = pd.concat([time_df, pd.Series([-1, call_starttime, timedelta(0), timedelta(0), timedelta(0)],
                                            index=["k", "time", "suggest", "target", "fit"])],
                        ignore_index=True)
    return time_df


def save_time_df(time_df, json_dict):
    def convert_time_to_float(x):
        try:
            y = x.seconds + x.microseconds / 1000000
        except AttributeError:
            y = x
        return y

    time_df.loc[:, "suggest"] = time_df.loc[:, "suggest"].apply(convert_time_to_float)
    time_df.loc[:, "target"] = time_df.loc[:, "target"].apply(convert_time_to_float)
    time_df.loc[:, "fit"] = time_df.loc[:, "fit"].apply(convert_time_to_float)
    time_df.to_csv(json_dict.time_file_path, sep=";")

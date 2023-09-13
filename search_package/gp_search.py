import json
from copy import deepcopy
from shutil import rmtree
from sys import gettrace

import warnings
import numpy as np
import pandas as pd
import scipy as sp
import os
from os.path import join
from addict import Dict
from joblib import Parallel, delayed
from datetime import datetime, timedelta
import traceback
import logging

from search_package.bayes_opt_custom import BayesianOptimizationMulti

from search_package.gp_funcs import create_kernel, create_gp_list, GaussianListWrapper, \
    create_target_function, search_step
from search_package.helper_functions import turn_experiment_list_to_dict, \
    turn_parameter_list_to_dict, compile_feature_score_names_list, init_directories, \
    bounds_dict_from_json, init_target_functions_and_sims, calculate_mask, load_previous_results, \
    init_in_bounds, collect_res_space, plot, check_boundaries, load_time_df, \
    save_time_df, divide_weights_by_scorenumber
from search_package.scores import init_score_dict
from search_package.base_classes import mean_agg_func, var_agg_func
from search_package.cadet_interface import plot_sim


def search_loop(meta_dict, close_figures=True, first_start=False, **kwargs):
    try:
        call_starttime = datetime.now()
        if first_start:
            meta_dict.starttime = str(call_starttime)
        warnings.filterwarnings('ignore')

        """ Initialization of metadata """

        if "termination_threshold" not in kwargs:
            kwargs["termination_threshold"] = 1e-4
        if "termination_count" not in kwargs:
            kwargs["termination_count"] = 250
        if "stall_threshold" not in kwargs:
            kwargs["stall_threshold"] = 0.0001
        if "stall_iterations" not in kwargs:
            kwargs["stall_iterations"] = 10

        # multiply search kwargs with number of dimensions
        kwargs.update(Dict({key: val * len(meta_dict.parameters)
                            for key, val in kwargs.items() if "threshold" not in key and "kappa" not in key}))
        meta_dict.update(kwargs)
        meta_dict.warmup_points = int(kwargs["warmup"])

        # create names for directories
        print(f"starting {meta_dict.baseDir, meta_dict.resultsDir}")
        meta_dict["fig_base_path"] = join(meta_dict.baseDir, meta_dict.resultsDir, "figures")
        meta_dict.sim_tmp_base_path = join(meta_dict.baseDir, meta_dict.resultsDir, "sim_tmp_storage")
        meta_dict.sim_source_base_path = join(meta_dict.baseDir, meta_dict.resultsDir, "sim_source_storage")
        meta_dict["time_file_path"] = join(meta_dict.baseDir, meta_dict.resultsDir, "time.csv")

        score_dict = init_score_dict()

        meta_dict = turn_parameter_list_to_dict(meta_dict)
        meta_dict = divide_weights_by_scorenumber(meta_dict)
        meta_dict = turn_experiment_list_to_dict(meta_dict)
        meta_dict = compile_feature_score_names_list(meta_dict, score_dict)
        json_file_path = join(meta_dict.baseDir, meta_dict.resultsDir, "meta.json")

        if os.path.exists(json_file_path):
            with open(json_file_path) as handle:
                meta_dict = Dict(json.load(handle))
        else:
            init_directories(meta_dict)

        if os.path.exists(meta_dict.time_file_path):
            time_df = load_time_df(meta_dict, call_starttime)
        else:
            time_df = pd.DataFrame(
                [-1, call_starttime, timedelta(0), timedelta(0), timedelta(0)],
                index=["k", "time", "suggest", "target", "fit"]).T
            save_time_df(time_df, meta_dict)

        if "n_jobs" not in meta_dict:
            meta_dict["n_jobs"] = 1

        logger = logging.getLogger(meta_dict.resultsDir)
        if first_start:
            f_handler = logging.FileHandler(join(meta_dict.baseDir,
                                                 meta_dict.resultsDir,
                                                 "log.log"))
            formatter = logging.Formatter('%(asctime)s [0] %(levelname)s - tmp - %(message)s')
            f_handler.setFormatter(formatter)
            logger.addHandler(f_handler)
            logger.setLevel(logging.INFO)
            logger.info(f"starting file {meta_dict.baseDir, meta_dict.resultsDir}")

        bounds_dict = bounds_dict_from_json(meta_dict)
        bounds_array = np.array([val for key, val in sorted(bounds_dict.items())])

        sim_templates, target_function_list = init_target_functions_and_sims(meta_dict, score_dict)

        gp_list, gp_name_list = create_gp_list(meta_dict, kernel=create_kernel)
        mask = calculate_mask(meta_dict)

        gps = GaussianListWrapper(gp_list, mean_agg_func, var_agg_func, bounds=bounds_dict,
                                  mask=mask, filter_x_to_unit_cube=False,
                                  transform=True, use_internal_parallelization=False,
                                  use_uncertainty_as_weights=False)

        target_function = create_target_function(target_function_list, mean_agg_func, inner_njobs=1,
                                                 only_aggregate=meta_dict.only_aggregate_score)

        optimizer = BayesianOptimizationMulti(f=target_function, pbounds=bounds_dict, gp=gps)
        # utility = UtilityFunction(kind="ucb", kappa=2.5, xi=-0.2)

        results_csv_file = join(meta_dict.baseDir, meta_dict.resultsDir, meta_dict.CSV)

        """ Initialization of algorithm - i.e. initial set of scattered points """

        k = 1
        _res_space = None
        initialization_necessary = True
        if os.path.exists(results_csv_file):
            initialization_necessary = False
            optimizer, _res_space = load_previous_results(optimizer, results_csv_file,
                                                          bounds_dict, index_limit=None,
                                                          return_res_space=True)
            optimizer.fit_gp()
            k = _res_space.index.max()
            if "only_plot" in kwargs and kwargs["only_plot"]:
                # if True:
                print("creating plots")
                _res_space = _res_space.sort_values("st", ascending=True)
                next_point = dict(_res_space.iloc[0, :len(meta_dict.parameters)])
                uuid = _res_space.index[0]
                _ = target_function(uuid=str(uuid), best_score=optimizer.min["target"],
                                    **next_point)
                plot_sim(str(uuid), meta_dict, plot_deriv=True)
                plot(k, optimizer, gps, bounds_dict, gp_name_list, meta_dict,
                     slice_point=optimizer.min["params"], close_figures=close_figures)
                rmtree(meta_dict.sim_tmp_base_path)
                os.makedirs(meta_dict.sim_tmp_base_path)
                return meta_dict, False

            # noinspection PyProtectedMember
            remaining_x = optimizer._gp.filtered_x_shape[0]
            if remaining_x is not None and remaining_x < meta_dict["warmup_points"]:
                initialization_necessary = True
                message = f"Only {remaining_x} samples remain in search space after" \
                          " dimension trunaction. Cueing reinitialization with LHS."
                print(message)
                logger.info(message)
        if initialization_necessary:
            optimizer = init_in_bounds(meta_dict, bounds_array, bounds_dict,
                                       target_function, optimizer, factorial=False)

            _res_space = collect_res_space(optimizer, bounds_dict, gp_name_list,
                                           results_csv_file)
            optimizer.fit_gp()
            k = _res_space.index.max() + 1
            time_df = pd.concat([time_df, pd.Series([k, datetime.now(), 0, 0, 0],
                                                    index=["k", "time", "suggest", "target", "fit"])],
                                ignore_index=True)
            save_time_df(time_df, meta_dict)
            if meta_dict.create_plots:
                plot(k, optimizer, gps, bounds_dict, gp_name_list, meta_dict,
                     slice_point=optimizer.min["params"], close_figures=close_figures)
            time_df = pd.concat([time_df, pd.Series([k + 0.5, datetime.now(), 0, 0, 0],
                                                    index=["k", "time", "suggest", "target", "fit"])],
                                ignore_index=True)
            save_time_df(time_df, meta_dict)


        if kwargs["run_least_squares"]:
            best_params = _res_space.iloc[0, :len(bounds_dict)].values
            least_sqr_results = sp.optimize.least_squares(target_function, best_params, bounds=bounds_array)

        """ Stepwise search"""

        if "search_kappa" in kwargs:
            search_kappa_list = np.linspace(kwargs["search_kappa"], 0, int(kwargs["search"]))
        else:
            search_kappa_list = [0.3] * int(kwargs["search"])

        for kappa in search_kappa_list:
            optimizer, time_df = search_step(k, kappa, optimizer, target_function, time_df, meta_dict)
            _res_space = collect_res_space(optimizer, bounds_dict, gp_name_list, results_csv_file)
            if check_if_score_threshold_met(_res_space,
                                            threshold=kwargs["termination_threshold"],
                                            stall_threshold=kwargs["stall_threshold"],
                                            stall_iterations=kwargs["stall_iterations"],
                                            logger=logger) \
                    or k > kwargs["termination_count"]:
                if meta_dict.create_plots:
                    plot(k, optimizer, gps, bounds_dict, gp_name_list, meta_dict,
                         slice_point=optimizer.min["params"], close_figures=close_figures)
                end_of_search(sim_templates, logger, meta_dict, call_starttime)
                return meta_dict, False
            k += 1


        """ Preparation for next iteration """

        if meta_dict.create_plots:
            plot(k, optimizer, gps, bounds_dict, gp_name_list, meta_dict,
                 slice_point=optimizer.min["params"], close_figures=close_figures)
            time_df = pd.concat([time_df, pd.Series([k + 0.5, datetime.now(), 0, 0, 0],
                                                    index=["k", "time", "suggest", "target", "fit"])],
                                ignore_index=True)
        save_time_df(time_df, meta_dict)

        meta_dict = check_boundaries(meta_dict, _res_space,
                                     score_threshold_count=int(kwargs["thresh_c"]))
        # this dumps the json file and creates a "log" copy of the json
        init_directories(meta_dict)
        logger.info("restarting search with updated boundaries")
        time_df = pd.concat([time_df, pd.Series([-2, datetime.now(), 0, 0, 0],
                                                index=["k", "time", "suggest", "target", "fit"])],
                            ignore_index=True)
        save_time_df(time_df, meta_dict)
        rmtree(meta_dict.sim_tmp_base_path)
        os.makedirs(meta_dict.sim_tmp_base_path)
        return meta_dict, True
    except Exception:
        try:
            print("\r\n\r\n")
            print(f"****** ERROR in file {meta_dict.baseDir, meta_dict.resultsDir} ****** @ {datetime.now()}")
            traceback.print_exc()
            print("\r\n\r\n")

            logger.debug("\r\n\r\n")
            logger.debug(f"****** ERROR in file {meta_dict.baseDir, meta_dict.resultsDir} ****** @ {datetime.now()}")
            logger.debug(traceback.format_exc())
            logger.debug("\r\n\r\n")

            rmtree(meta_dict.sim_tmp_base_path)
            os.makedirs(meta_dict.sim_tmp_base_path)

            # noinspection PyUnboundLocalVariable
            _res_space = collect_res_space(optimizer, bounds_dict, gp_name_list,
                                           results_csv_file)
        except NameError:
            pass
        return meta_dict, False


def end_of_search(sim_templates, logger, meta_dict, call_starttime):
    for template_sim in sim_templates:
        os.remove(template_sim.filename)
    logger.info(f"{meta_dict.baseDir, meta_dict.resultsDir} TOOK {datetime.now() - call_starttime}")
    rmtree(meta_dict.sim_tmp_base_path)
    os.makedirs(meta_dict.sim_tmp_base_path)


def check_if_score_threshold_met(_res_space, threshold=0.005, stall_threshold=0.0001, stall_iterations=10 * 3,
                                 logger=None):
    res_space_index_sorted = _res_space.sort_index()
    best_params = _res_space.iloc[0, :]
    global_threshold_met = best_params.st <= threshold
    if _res_space.shape[0] > stall_iterations:
        improvement_over_x_iterations = res_space_index_sorted.iloc[:-stall_iterations, :].st.min() - best_params.st
    else:
        improvement_over_x_iterations = res_space_index_sorted.iloc[0, :].st - best_params.st
    stall_threshold_met = improvement_over_x_iterations < stall_threshold
    stall_buffer_exceeded = _res_space.shape[0] > stall_iterations
    if global_threshold_met:
        if logger is not None:
            logger.info(f"Termination condition global_threshold of {threshold} "
                        f"was reached with value {best_params.st}.")
        return True
    elif stall_threshold_met and stall_buffer_exceeded:
        if logger is not None:
            logger.info(f"Termination condition stall_threshold of "
                        f"less than {stall_threshold} improvement over {stall_iterations} tries was reached "
                        f"with {best_params.st} improvement after {_res_space.shape[0]} total.")
        return True
    else:
        return False


def search(config_file_path):
    logging.basicConfig(format='%(asctime)s [0] %(levelname)s - tmp - %(message)s', level=0)
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(logging.WARNING)
    with open(config_file_path) as handle:
        meta_dict = Dict(json.load(handle))
    search_kwargs = meta_dict.search_kwargs_input

    meta_dict, rerun_needed = search_loop(meta_dict,
                                          first_start=True,
                                          **search_kwargs)

    for _ in range(15):
        if rerun_needed:
            meta_dict, rerun_needed = search_loop(meta_dict,
                                                  first_start=False,
                                                  **search_kwargs)

    return


if __name__ == "__main__":
    config_file_path = "path/to/config_file.json"
    search(config_file_path)

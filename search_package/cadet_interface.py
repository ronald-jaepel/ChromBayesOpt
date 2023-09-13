import shutil
import tempfile
import os
import traceback
from os.path import join
import subprocess
import pprint
import copy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from addict import Dict
from copy import deepcopy
from logging import getLogger
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py

from search_package.scores import find_peak
from search_package.base_classes import transforms_dict


class Cadet:
    # cadet_path must be set in order for simulations to run
    cadet_path = None
    return_information = None

    pp = pprint.PrettyPrinter(indent=4)

    def __init__(self, *data):
        self.root = Dict()
        for i in data:
            self.root.update(copy.deepcopy(i))

    def load(self):
        with h5py.File(self.filename, 'r') as h5file:
            self.root = Dict(recursively_load(h5file, '/'))

    def save(self):
        with h5py.File(self.filename, 'w') as h5file:
            recursively_save(h5file, '/', self.root)

    def run(self, timeout=None, check=None):

        if not os.path.exists(self.cadet_path):
            raise ValueError("cadet_path points to a missing file.")

        data = subprocess.run([self.cadet_path, self.filename], timeout=timeout, check=check)  # , capture_output=True)
        self.return_information = data
        if data.stderr:
            print(data.stderr)
            raise RuntimeError(data.stdout)
        return data

    def __str__(self):
        temp = []
        temp.append('Filename = %s' % self.filename)
        temp.append(self.pp.pformat(self.root))
        return '\n'.join(temp)

    def update(self, merge):
        self.root.update(merge.root)

    def __getitem__(self, key):
        key = key.lower()
        obj = self.root
        for i in key.split('/'):
            if i:
                obj = obj[i]
        return obj

    def __setitem__(self, key, value):
        key = key.lower()
        obj = self.root
        parts = key.split('/')
        for i in parts[:-1]:
            if i:
                obj = obj[i]
        obj[parts[-1]] = value


def recursively_load(h5file, path):
    ans = {}
    for key, item in h5file[path].items():
        key = key.lower()
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load(h5file, path + key + '/')
    return ans


def recursively_save(h5file, path, dic):
    # argument type checking
    if not isinstance(dic, dict):
        raise ValueError("must provide a dictionary")

    if not isinstance(path, str):
        raise ValueError("path must be a string")
    if not isinstance(h5file, h5py._hl.files.File):
        raise ValueError("must be an open h5py file")
    # save items to the hdf5 file
    for key, item in dic.items():
        key = str(key)
        if not isinstance(key, str):
            raise ValueError("dict keys must be strings to save to hdf5")
        # handle   int, float, string and ndarray of int32, int64, float64
        if isinstance(item, str):
            h5file[path + key.upper()] = np.array(item, dtype='S')

        elif isinstance(item, int):
            h5file[path + key.upper()] = np.array(item, dtype=np.int32)

        elif isinstance(item, float):
            h5file[path + key.upper()] = np.array(item, dtype=np.float64)

        elif isinstance(item, np.ndarray) and item.dtype == np.float64:
            h5file[path + key.upper()] = item

        elif isinstance(item, np.ndarray) and item.dtype == np.float32:
            h5file[path + key.upper()] = np.array(item, dtype=np.float64)

        elif isinstance(item, np.ndarray) and item.dtype == np.int32:
            h5file[path + key.upper()] = item

        elif isinstance(item, np.ndarray) and item.dtype == np.int64:
            h5file[path + key.upper()] = item.astype(np.int32)

        elif isinstance(item, np.ndarray) and item.dtype.kind == 'S':
            h5file[path + key.upper()] = item

        elif isinstance(item, list) and all(isinstance(i, int) for i in item):
            h5file[path + key.upper()] = np.array(item, dtype=np.int32)

        elif isinstance(item, list) and any(isinstance(i, float) for i in item):
            h5file[path + key.upper()] = np.array(item, dtype=np.float64)

        elif isinstance(item, np.int32):
            h5file[path + key.upper()] = item

        elif isinstance(item, np.float64):
            h5file[path + key.upper()] = item

        elif isinstance(item, np.float32):
            h5file[path + key.upper()] = np.array(item, dtype=np.float64)

        elif isinstance(item, np.bytes_):
            h5file[path + key.upper()] = item

        elif isinstance(item, bytes):
            h5file[path + key.upper()] = item

        elif isinstance(item, list) and all(isinstance(i, str) for i in item):
            h5file[path + key.upper()] = np.array(item, dtype="S")

        # save dictionaries
        elif isinstance(item, dict):
            recursively_save(h5file, path + key + '/', item)
        # other types cannot be saved and will result in an error
        else:
            raise ValueError('Cannot save %s/%s key with %s type.' % (path, key.upper(), type(item)))


def setup_template(template):
    try:
        del template.root.output
    except KeyError:
        pass
    template.root.input['return'].unit_001.write_solution_particle = 0
    template.root.input['return'].unit_001.write_solution_column_inlet = 1
    template.root.input['return'].unit_001.write_solution_column_outlet = 1
    template.root.input['return'].unit_001.write_solution_inlet = 1
    template.root.input['return'].unit_001.write_solution_outlet = 1
    template.root.input['return'].unit_001.split_components_data = 0
    template.root.input.solver.nthreads = 1
    template.save()
    return template


def run_cadet(exp_name, sim_file_name, template_sim, json_dict):
    simulation = Cadet(template_sim.root)
    simulation.cadet_path = json_dict["CADETPath"]
    simulation.filename = sim_file_name
    # print(sim_file_name, exp_name)

    for param_name, parameter in json_dict.parameters.items():
        if any((exp_name in target_exp or "just_set" in target_exp)
               for target_exp in parameter.experiments) \
                and "value" in parameter:
            param_dtype = type(simulation[parameter.location])
            # print(f"setting {param_name} to {parameter.value}")
            if parameter.component == -1:
                if param_dtype == np.ndarray:
                    simulation[parameter.location] = np.ones_like(
                        simulation[parameter.location]) * parameter.value
                else:
                    simulation[parameter.location] = parameter.value
            else:
                if param_dtype != np.ndarray:
                    raise ValueError(
                        f"Component specified for paramter that has no component specific values\n{parameter}")
                simulation[parameter.location][parameter.component] = parameter.value

    simulation.save()

    logger = getLogger(json_dict.resultsDir)
    # logger.info(f"ran sim with {collect_param_values(json_dict)}")

    try:
        simulation.run(timeout=json_dict["timeout"], check=True)
    except subprocess.TimeoutExpired:
        error_message = 'Simulation Timed Out with \r\n' + collect_param_values(json_dict)
        logger.debug(error_message)
        print(error_message)
        return None
    except subprocess.CalledProcessError as error:
        error_message = 'Simulation Failed with \r\n' + collect_param_values(json_dict) \
                        + f"\r\n error message: \r\n {error}"
        logger.error(error_message)
        print(error_message)
        return None

    # read sim data
    simulation.load()

    simulation_failed = isinstance(simulation.root.output.solution.solution_times, Dict)
    if simulation_failed:
        print("sim must have failed")
        logger.debug("Simulation Returned without solution")
        logger.debug("sim parameters were:")
        logger.debug(str(json_dict.parameters.items()))
        return None

    return simulation


def collect_param_values(json_dict):
    message = ""
    for param_name, param in json_dict.parameters.items():
        if "value" in param and "transform" in param:
            transformed_val = transforms_dict[param.transform + "_inv"](param.value)
            message += f"{param_name}: {round(transformed_val[0], 2)}\t "
    return message


def find_squarest_arrangement(n, shape_per_box=(1, 1)):
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
    if n == 1:
        smallest_cols = 1
        smallest_row = 1
    return int(smallest_row), int(smallest_cols)


def prepare_and_run_cadet(uuid, exp_name, template, datatargets, json_dict, score_dict, best_score):
    json_dict["return_sim"] = False

    unit = template.root.input.model.unit_001
    if "cv" in exp_name:
        time_offset_before = (unit.col_porosity + (1 - unit.col_porosity) * unit.par_porosity) \
                             / 0.5 * 60 + 0.6

    filename = os.path.join(f'{json_dict["sim_tmp_base_path"]}', f'{uuid}_{exp_name}.h5')
    try:
        sim = run_cadet(exp_name, filename, template, json_dict)
    except OSError as e:
        print(e)
        traceback.print_exc()
        raise OSError(exp_name, filename)

    if sim is None:
        return [np.nan if json_dict["using_nans"] else -2
                for feature_name, feature in json_dict.experiments[exp_name].features.items()
                for _ in range(len(score_dict[feature.type][1]))]

    unit = sim.root.input.model.unit_001
    if "cv" in exp_name:
        time_offset_after = (unit.col_porosity + (1 - unit.col_porosity) * unit.par_porosity) \
                            / 0.5 * 60 + 0.6
        json_dict["time_offset_delta"] = time_offset_before - time_offset_after
        sim.root.meta.time_offset_delta = json_dict["time_offset_delta"]
    if "test_bill" in str(uuid):
        json_dict["time_offset_delta"] = 0

    scores = []
    score_batches = {}

    features = json_dict.experiments[exp_name].features

    for feature_name, feature in sorted(features.items()):
        if "decay" in feature:
            json_dict["time_tolerance"] = feature.decay
        else:
            json_dict["time_tolerance"] = 0

        sim_results = sim[feature.isotherm]

        target = datatargets[feature.CSV]

        time_selection = get_time_selection(feature, target.time.values, target.signal.values)

        sim_time_values = sim.root.output.solution.solution_times
        if "cv" in feature_name:
            try:
                sim_time_values = sim_time_values + json_dict["time_offset_delta"]
            except:
                raise RuntimeError("Trying to adjust 'time_offset' in gradient elution, "
                                   "but time offset was not calculated. Do the experiments"
                                   " with 'cv' in their feature name also have 'cv' in their experiment name?")

        score = score_dict[feature.type][0](target.time.values, target.signal.values,
                                            sim_time_values, sim_results, time_selection, json_dict)
        if "weights" in feature:
            try:
                score *= np.array(feature.weights)
            except ValueError:
                pass
        scores.extend(score)
        score_batches[feature_name] = score
        sim.root.meta.data_targets[feature_name] = target.values

    if features == {}:
        experiment = json_dict.experiments[exp_name]
        sim.root.meta.data_targets[experiment.CSV] = datatargets[experiment.CSV].values

    sim.root.meta.score_batches = score_batches
    sim.save()
    return scores


def get_time_selection(feature, target_x, target_y):
    if "time_selection" in feature:
        if feature["time_selection"] == "peak_max":
            peak_max_time, peak_max_value = find_peak(target_x, target_y)[0]
            time_selection = (0, peak_max_time)
        else:
            time_selection = feature["time_selection"]
    else:
        time_selection = (0, 1e100)
    return time_selection


def plot_sim(uuid, json_dict, plot_deriv=False):
    plt.ioff()
    plots_required = 0
    for experiment in json_dict.experiments.values():
        if all("CSV" in feature for feature in experiment.features.values()) \
                and len(set([feature["CSV"] for feature in experiment.features.values()])) > 1:
            plots_required += len(experiment.features)
        else:
            plots_required += 1

    if plot_deriv:
        cols, rows = find_squarest_arrangement(plots_required * 2)
    else:
        cols, rows = find_squarest_arrangement(plots_required)

    i = -1
    fig, axes = plt.subplots(cols, rows, figsize=(5 * rows, 4 * cols))
    try:
        axes = axes.flatten()
    except AttributeError:
        axes = [axes]
    try:
        for exp_name, experiment in json_dict.experiments.items():
            features = experiment.features
            if not (all("CSV" in feature for feature in experiment.features.values())
                    and len(set([feature["CSV"] for feature in experiment.features.values()])) > 1):
                feature_name, feature = list(sorted(features.items()))[0]
                features = Dict({feature_name: feature})
            if features == {}:
                features = Dict({experiment.CSV: {"isotherm": experiment.isotherm}})

            for feature_name, feature in sorted(features.items()):
                i += 1
                ax = axes[i]
                sim = Cadet()
                sim.filename = f'{json_dict["sim_tmp_base_path"]}/{uuid}_{exp_name}.h5'
                sim.load()
                # sim.filename = f'{json_dict["fig_base_path"]}_sims/{uuid}_{exp_name}.h5'
                # sim.save()

                sim_results = sim[feature.isotherm]
                target = sim.root.meta.data_targets[feature_name]

                sim_time_values = sim.root.output.solution.solution_times
                if "cv" in feature_name and sim.root.meta.time_offset_delta != {}:
                    sim_time_values = sim_time_values + sim.root.meta.time_offset_delta

                # time_selection = get_time_selection(feature, target[:, 0], target[:, 1])
                time_selection = 0, 1e12
                sim_results = sim_results[(time_selection[0] < sim_time_values) &
                                          (sim_time_values < time_selection[1])]
                sim_time_values = sim_time_values[(time_selection[0] < sim_time_values) &
                                                  (sim_time_values < time_selection[1])]

                target_signal = target[:, 1]
                target_time = target[:, 0]
                target_signal = target_signal[(time_selection[0] < target_time) &
                                              (target_time < time_selection[1])]
                target_time = target_time[(time_selection[0] < target_time) &
                                          (target_time < time_selection[1])]
                ax.plot(target_time, target_signal, label="exp")
                ax.plot(sim_time_values, sim_results, label="sim")
                ax2 = ax.twinx()
                if "cv" in feature_name:
                    ax2.plot(sim.root.output.solution.solution_times,
                             sim.root.output.solution.unit_001.solution_outlet_comp_000,
                             '--', label="salt")
                experiment_scores = [f"{fn}: {np.round(sim.root.meta.score_batches[fn], 3)}"
                                     for fn, _ in sorted(experiment.features.items())]
                ax.set_title(" ".join(experiment_scores))
                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines + lines2, labels + labels2, loc=0)
                if plot_deriv:
                    dif_target1 = np.diff(target_signal, n=1)
                    dif1 = np.diff(sim_results, n=1)
                    ax_2 = axes[i + len(json_dict.experiments)]
                    ax_2.plot(target_time[:-1], dif_target1, label="exp")
                    ax_2.plot(sim_time_values[:-1], dif1, label="sim")
                    ax_2.legend()

        fig.tight_layout()
        fig.savefig(f'{json_dict["fig_base_path"]}_sims/{uuid}.png', dpi=200)
        plt.close(fig)
    except:
        traceback.print_exc()
    return


def load_template_sim(h5_filename, root_dir, cadet_path, tmp_dir=None):
    h5_path = join(root_dir, h5_filename)
    if tmp_dir is not None and len(tmp_dir) != 0:
        handle, path = tempfile.mkstemp(suffix='.h5', dir=tmp_dir)
    else:
        handle, path = tempfile.mkstemp(suffix='.h5')
    os.close(handle)
    shutil.copyfile(h5_path, path)

    template_sim = Cadet()
    template_sim.cadet_path = cadet_path
    template_sim.filename = path
    template_sim.load()
    template_sim = setup_template(template_sim)
    return template_sim


def load_data_target(data_target_filename, root_dir):
    data_target_path = join(root_dir, data_target_filename)
    with open(data_target_path, "r") as file_handle:
        if ";" in file_handle.readline():
            sep = ";"
        else:
            sep = ","
    data_target = pd.read_csv(data_target_path, header=None, names=["time", "signal"], sep=sep)
    return data_target


def preapply_exp_info(exp_name, template, datatarget, json_dict, score_dict):
    def wrapper_function(uuid, best_score=999, **kwargs_input):
        json_dict_internal = deepcopy(json_dict)
        # set kwags parameters into internal json_dict
        for par, val in kwargs_input.items():
            if par == "TOT_POROSITY":
                if "COL_POROSITY" in kwargs_input:
                    col_transform = json_dict_internal["parameters"]["COL_POROSITY"]["transform"]
                    col_poros = transforms_dict[col_transform](kwargs_input["COL_POROSITY"])
                    parameter_transform = json_dict_internal["parameters"][par]["transform"]
                    val = transforms_dict[parameter_transform](val)
                    par_poros = (val - col_poros) / (1 - col_poros)
                    json_dict_internal["parameters"]["PAR_POROSITY"]["value"] = par_poros
                elif "PAR_POROSITY" in kwargs_input:
                    par_transform = json_dict_internal["parameters"]["PAR_POROSITY"]["transform"]
                    par_poros = transforms_dict[par_transform](kwargs_input["PAR_POROSITY"])
                    parameter_transform = json_dict_internal["parameters"][par]["transform"]
                    val = transforms_dict[parameter_transform](val)
                    col_poros = (val - par_poros) / (1 - par_poros)
                    json_dict_internal["parameters"]["COL_POROSITY"]["value"] = col_poros
                else:
                    raise ValueError("TOT_POROSITY must be set with either PAR_POROSITY or "
                                     "COL_POROSITY")

            if "KEQ" in par:
                isotherm_prefix = par.split("KEQ")[0]
                try:
                    component = par.split("_c")[-1]
                except Exception as e:
                    raise Exception("KEQ was given without a proper component") from e
                ka_name = f"{isotherm_prefix}KA_c{component}"
                kd_name = f"{isotherm_prefix}KD_c{component}"
                if any(ka_name in kwarg for kwarg in kwargs_input):
                    ka_transform = json_dict_internal["parameters"][ka_name]["transform"]
                    ka = transforms_dict[ka_transform](kwargs_input[ka_name])
                    keq_transform = json_dict_internal["parameters"][par]["transform"]
                    keq = transforms_dict[keq_transform](val)
                    kd = ka / keq
                    json_dict_internal["parameters"][kd_name] = \
                        deepcopy(json_dict_internal["extra_parameter_infos"][kd_name])
                    json_dict_internal["parameters"][kd_name]["value"] = kd
                elif any(kd_name in kwarg for kwarg in kwargs_input):
                    kd_transform = json_dict_internal["parameters"][kd_name]["transform"]
                    kd = transforms_dict[kd_transform](kwargs_input[kd_name])
                    keq_transform = json_dict_internal["parameters"][par]["transform"]
                    keq = transforms_dict[keq_transform](val)
                    ka = kd * keq
                    json_dict_internal["parameters"][ka_name] = \
                        deepcopy(json_dict_internal["extra_parameter_infos"][ka_name])
                    json_dict_internal["parameters"][ka_name]["value"] = ka
                else:
                    keq_transform = json_dict_internal["parameters"][par]["transform"]
                    keq = transforms_dict[keq_transform](val)
                    kd = 1e3 / keq
                    json_dict_internal["parameters"][kd_name] = \
                        deepcopy(json_dict_internal["extra_parameter_infos"][kd_name])
                    json_dict_internal["parameters"][kd_name]["value"] = kd
                    json_dict_internal["parameters"][ka_name] = \
                        deepcopy(json_dict_internal["extra_parameter_infos"][kd_name])
                    json_dict_internal["parameters"][ka_name]["location"] = \
                        json_dict_internal["parameters"][ka_name]["location"].replace("KD", "KA")
                    json_dict_internal["parameters"][ka_name]["value"] = 1e3
            else:
                parameter_transform = json_dict_internal["parameters"][par]["transform"]
                val = transforms_dict[parameter_transform](val)
                json_dict_internal["parameters"][par]["value"] = val

        # used to store sims in sim_storage
        # experimental feature
        space_descriptor = str(
            np.round(np.concatenate(list(kwargs_input.values()), axis=-1), 4)).replace("  ", " ")
        json_dict_internal["fig_path"] = join(json_dict_internal["fig_base_path"],
                                              space_descriptor + "_" + exp_name + ".png")

        score_results = prepare_and_run_cadet(uuid, exp_name, template, datatarget,
                                              json_dict_internal,
                                              score_dict, best_score)
        """Reshape your data either using array.reshape(-1, 1) if
        your data has a single feature or array.reshape(1, -1)
        if it contains a single sample."""
        return np.array(score_results).reshape(1, -1)

    return wrapper_function

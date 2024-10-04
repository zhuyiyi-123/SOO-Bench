import os
import json
import torch
from typing import Dict
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from ray.tune.logger import Logger, CSVLogger, JsonLogger, VALID_SUMMARY_TYPES
from ray.tune.utils import flatten_dict
from ray.tune.error import TuneError
from ray.tune import Stopper


class SysStopper(Stopper):
    def __init__(self, workspace, max_iter: int = 0, stop_callback = None):
        self._workspace = workspace
        self._max_iter = max_iter
        self._iter = defaultdict(lambda: 0)
        self.stop_callback = stop_callback

    def __call__(self, trial_id, result):
        if self._max_iter > 0:
            self._iter[trial_id] += 1
            if self._iter[trial_id] >= self._max_iter:
                return True
        if result["stop_flag"]:
            if self.stop_callback:
                self.stop_callback()
            return True
        
        return False

    def stop_all(self):
        if os.path.exists(os.path.join(self._workspace,'.env.json')):
            with open(os.path.join(self._workspace,'.env.json'), 'r') as f:
                _data = json.load(f)
            if _data["REVIVE_STOP"]:
                if self.stop_callback:
                    self.stop_callback()
            return _data["REVIVE_STOP"]
        else:
            return False

class TuneTBLogger(Logger):
    r"""
        custom tensorboard logger for ray tune
        modified from ray.tune.logger.TBXLogger
    """
    def _init(self):
        self._file_writer = SummaryWriter(self.logdir)
        self.last_result = None
        self.step = 0

    def on_result(self, result):
        self.step += 1

        tmp = result.copy()
        flat_result = flatten_dict(tmp, delimiter="/")

        for k, v in flat_result.items():
            if type(v) in VALID_SUMMARY_TYPES:
                self._file_writer.add_scalar(k, float(v), global_step=self.step)
            elif isinstance(v, torch.Tensor):
                v = v.view(-1)
                self._file_writer.add_histogram(k, v, global_step=self.step)

        self.last_result = flat_result
        self.flush()

    def flush(self):
        if self._file_writer is not None:
            self._file_writer.flush()

TUNE_LOGGERS = (CSVLogger, JsonLogger, TuneTBLogger)

import logging, copy
from ray.tune.suggest import BasicVariantGenerator, SearchGenerator, Searcher
from ray.tune.config_parser import create_trial_from_spec
from ray.tune.suggest.variant_generator import generate_variants, format_vars, resolve_nested_dict, flatten_resolved_vars
from ray.tune.trial import Trial
from ray.tune.utils import merge_dicts, flatten_dict
logger = logging.getLogger(__name__)

class CustomSearchGenerator(SearchGenerator):
    def create_trial_if_possible(self, experiment_spec, output_path):
        logger.debug("creating trial")
        trial_id = Trial.generate_id()
        suggested_config = self.searcher.suggest(trial_id)

        if suggested_config == Searcher.FINISHED:
            self._finished = True
            logger.debug("Searcher has finished.")
            return

        if suggested_config is None:
            return

        spec = copy.deepcopy(experiment_spec)
        spec["config"] = merge_dicts(spec["config"],
                                     copy.deepcopy(suggested_config))

        # Create a new trial_id if duplicate trial is created
        flattened_config = resolve_nested_dict(spec["config"])
        self._counter += 1
        tag = "{0}_{1}".format(
            str(self._counter), format_vars(flattened_config))
        spec['config']['tag'] = tag # pass down the tag
        trial = create_trial_from_spec(
            spec,
            output_path,
            self._parser,
            evaluated_params=flatten_dict(suggested_config),
            experiment_tag=tag,
            trial_id=trial_id)
        return trial

from ray.tune.suggest.basic_variant import _TrialIterator, convert_to_experiment_list, count_spec_samples, count_variants
from ray.tune.suggest.basic_variant import warnings, Union, List, itertools, Experiment, SERIALIZATION_THRESHOLD

class TrialIterator(_TrialIterator):
    def create_trial(self, resolved_vars, spec):
        trial_id = self.uuid_prefix + ("%05d" % self.counter)
        experiment_tag = str(self.counter)
        # Always append resolved vars to experiment tag?
        if resolved_vars:
            experiment_tag += "_{}".format(format_vars(resolved_vars))
        spec['config']['tag'] = experiment_tag
        self.counter += 1
        return create_trial_from_spec(
            spec,
            self.output_path,
            self.parser,
            evaluated_params=flatten_resolved_vars(resolved_vars),
            trial_id=trial_id,
            experiment_tag=experiment_tag)

class CustomBasicVariantGenerator(BasicVariantGenerator):
    def add_configurations(
            self,
            experiments: Union[Experiment, List[Experiment], Dict[str, Dict]]):
        """Chains generator given experiment specifications.

        Arguments:
            experiments (Experiment | list | dict): Experiments to run.
        """
        experiment_list = convert_to_experiment_list(experiments)
        for experiment in experiment_list:
            grid_vals = count_spec_samples(experiment.spec, num_samples=1)
            lazy_eval = grid_vals > SERIALIZATION_THRESHOLD
            if lazy_eval:
                warnings.warn(
                    f"The number of pre-generated samples ({grid_vals}) "
                    "exceeds the serialization threshold "
                    f"({int(SERIALIZATION_THRESHOLD)}). Resume ability is "
                    "disabled. To fix this, reduce the number of "
                    "dimensions/size of the provided grid search.")

            previous_samples = self._total_samples
            points_to_evaluate = copy.deepcopy(self._points_to_evaluate)
            self._total_samples += count_variants(experiment.spec,
                                                  points_to_evaluate)
            iterator = TrialIterator(
                uuid_prefix=self._uuid_prefix,
                num_samples=experiment.spec.get("num_samples", 1),
                unresolved_spec=experiment.spec,
                constant_grid_search=self._constant_grid_search,
                output_path=experiment.dir_name,
                points_to_evaluate=points_to_evaluate,
                lazy_eval=lazy_eval,
                start=previous_samples)
            self._iterators.append(iterator)
            self._trial_generator = itertools.chain(self._trial_generator,
                                                    iterator)

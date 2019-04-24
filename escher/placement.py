from collections import defaultdict

import ray
from ray.tune.trial import Trial
from ray.tune.schedulers import TrialScheduler
from ray.tune.schedulers import FIFOScheduler

import logging

logger = logging.getLogger(__name__)

class PlacementScheduler(FIFOScheduler):
    def __init__(self, consolidation_limit=4):
        self.limit = consolidation_limit
        self.all_placements = defaultdict(dict)
        self.all_live_trials = set()
        self.executor = None

    def on_trial_result(self, trial_runner, trial, result):
        if self.executor is None:
            self.executor = trial_runner.trial_executor
        self.all_live_trials.add(trial.trial_id)
        locations = result["locations"]  # This is a map from location -> count
        in_sync = self._track_trial(trial_runner, trial.trial_id, locations)
        if not in_sync:
            return TrialScheduler.CONTINUE
        else:
            logger.info(f"{trial} is in sync!")

        total_job_size = sum(locations.values())
        if total_job_size == 1 or total_job_size > self.limit:
            return TrialScheduler.CONTINUE
        for location, size in locations.items():
            rest_of_job = total_job_size - size
            if rest_of_job == 0:
                break
            print(f"All placements {self.all_placements}")
            print(f"{sum(self.all_placements[location].values()) + rest_of_job} <= {total_job_size} (limit)")
            if sum(self.all_placements[location].values()) + rest_of_job <= self.limit:
                self._migrate_trial(trial, total_job_size, location)
                break
        return TrialScheduler.CONTINUE

    def on_trial_complete(self, trial_runner, trial, result):
        self._track_trial(trial_runner, trial.trial_id, {t: 0 for t in result["locations"]})
        self.on_trial_remove(trial_runner, trial)

    def on_trial_remove(self, trial_runner, trial):
        self.all_live_trials.remove(trial.trial_id)

    def _track_trial(self, trial_runner, trial_id, locations):
        in_sync = True

        live_ids = [t.trial_id for t in trial_runner.get_trials() if t.status == Trial.RUNNING]
        if not len(live_ids) == len(self.all_live_trials):
            in_sync = False

        for path, count in list(locations.items()):
            location_trials = self.all_placements[path]
            if trial_id not in location_trials or location_trials[trial_id] != count:
                in_sync = False
                logger.warning(f"{trial_id} out of sync.")
                self.all_placements[path][trial_id] = count
        return in_sync

    def _migrate_trial(self, trial, total_job_size, location):
        self.executor.save(trial, storage="memory")
        self.executor.stop_trial(trial, stop_logger=False)
        trial.status = Trial.PENDING
        logger.warning("Trial stopped.")
        for other_location in self.all_placements:
            ray.experimental.delete_resource(trial.trial_id, other_location)
        logger.warning("Resources cleared.")
        # delete resources in all_location
        ray.experimental.create_resource(trial.trial_id, total_job_size, location)
        logger.warning("Resources created.")
        trial.resources.extra_custom_resources[trial.trial_id] = total_job_size
        trial.resources.custom_resources[trial.trial_id] = 0
        print(ray.global_state.client_table())
        print(ray.global_state.available_resources())
        print("New resources: {}".format(trial.resources.summary_string()))
        self.executor.start_trial(trial)

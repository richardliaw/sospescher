from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import time
import numpy as np
import random

import ray
from ray.tune.logger import NoopLogger
from ray.tune.ray_trial_executor import RayTrialExecutor
from ray.tune import Trainable

logger = logging.getLogger(__name__)


class ResourceExecutor(RayTrialExecutor):
    """An implemention of TrialExecutor based on Ray."""

    def _setup_runner(self, trial, reuse_allowed=False):
        cls = ray.remote(
            num_cpus=trial.resources.cpu,
            num_gpus=trial.resources.gpu,
            resources=trial.resources.custom_resources)(
                trial._get_trainable_cls())

        trial.init_logger()
        # We checkpoint metadata here to try mitigating logdir duplication
        self.try_checkpoint_metadata(trial)
        remote_logdir = trial.logdir

        def logger_creator(config):
            # Set the working dir in the remote process, for user file writes
            if not os.path.exists(remote_logdir):
                os.makedirs(remote_logdir)
            os.chdir(remote_logdir)
            return NoopLogger(config, remote_logdir)

        trial.config.setdefault("trial_id", trial.trial_id)

        # Logging for trials is handled centrally by TrialRunner, so
        # configure the remote runner to use a noop-logger.
        return cls.remote(
            config=trial.config,
            logger_creator=logger_creator,
            resources=trial.resources)


class ResourceTrainable(Trainable):
    def __init__(self, config=None, logger_creator=None, resources=None):
        self.resources = resources
        import os
        cwd = os.getcwd()
        super(ResourceTrainable, self).__init__(config, logger_creator)
        if ray.worker._mode() == ray.worker.LOCAL_MODE:
            os.chdir(cwd)


@ray.remote
class TestActor():
    def __init__(self):
        pass

    def get_client(self):
        client_table = ray.global_state.client_table()
        socket_name = str(ray.worker.global_worker.plasma_client.store_socket_name)
        return [k["ClientID"] for k in client_table if k["ObjectStoreSocketName"] == socket_name][0]


class Aggregator(ResourceTrainable):
    def _setup(self, config):
        # get throughput somehow
        self.trial_id = config["trial_id"]
        num_workers = self.resources.extra_cpu
        resources = None
        extra_res = self.resources.get_res_total(self.trial_id)
        if extra_res:
            assert extra_res == num_workers, "Trainable out of sync!"
            resources = {self.trial_id: 1}
        self.actors = []
        for i in range(num_workers):
            self.actors += [TestActor._remote(args=[], num_cpus=1, resources=resources)]

    def _train(self):
        import time
        time.sleep(self.config.get("delay", 0.1))
        from collections import Counter
        locations = Counter(ray.get([a.get_client.remote() for a in self.actors]))
        if len(locations) == 1:
            throughput = len(self.actors)
        else:
            throughput = 0.5
        random_stop = self.resources.extra_cpu * 10 + np.random.choice(np.r_[:20])
        return {
            "locations": dict(locations),
            "throughput": throughput,
            "done": self._iteration > random_stop
        }

    def _save(self, ckpt):
        return {}

    def _restore(self, ckpt):
        pass

    def _stop(self):
        for actor in self.actors:
            actor.__ray_terminate__.remote()

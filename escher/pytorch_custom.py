from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict, Counter
import math
import time
import sys
import logging
import tempfile
import torch
import torchvision
import os
import random
import pandas as pd
import numpy as np
import ray
from .util import TimerStat, create_colocated

import ray
from escher.distributed_trainable import ResourceTrainable

import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from .pytorch_helpers import (train, adjust_learning_rate, validate,
                              state_from_cuda, state_to_cuda)

DEFAULT_CONFIG = {
    # Arguments to pass to the optimizer
    "starting_lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    ## Change this to epoch units
    "steps_per_iteration": None,
    # Pins actors to cores
    "pin": False,
    "model_creator": None,
    "target_batch_size": 64,
    "model_string": "resnet101",
    "min_batch_size": 64,
    "dataset": "CIFAR",
    "placement": None,
    "use_nccl": False,
    # "num_workers": 2,
    "devices_per_worker": 1,
    "primary_resource": "extra_gpu",
    "gpu": False,
    "verbose": False
}

logger = logging.getLogger(__name__)


class PyTorchRunner(object):
    def __init__(self, batch_size, starting_lr=0.1, momentum=0.9,
                 weight_decay=5e-4, steps_per_iteration=None,
                 model_string="resnet18", verbose=False, use_nccl=False):
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        self.batch_size = batch_size
        self.epoch = 0
        self.starting_lr = starting_lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.use_nccl = use_nccl
        self.model_string = model_string
        self.verbose = verbose
        self._steps_per_iteration = steps_per_iteration
        self._timers = {
            "setup_proc": TimerStat(window_size=1),
            "setup_model": TimerStat(window_size=1),
            "get_state": TimerStat(window_size=1),
            "set_state": TimerStat(window_size=1),
            "validation": TimerStat(window_size=1),
            "training": TimerStat(window_size=1)
        }
        self.local_rank = None

    def set_device(self, original_cuda_id=None):
        self.local_rank = int(original_cuda_id or os.environ["CUDA_VISIBLE_DEVICES"])

    def setup_proc_group(self, dist_url, world_rank=0, world_size=1):
        # self.try_stop()
        self.world_rank = world_rank
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        with self._timers["setup_proc"]:

            ########## This is a hack because set_devices fails otherwise #################
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                [str(i) for i in range(ray.services._autodetect_num_gpus())])
            ################################################################################

            torch.cuda.set_device(self.local_rank)
            if self.verbose:
                logger.info("Device Set.")

    def setup_model(self):
        with self._timers["setup_model"]:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])  # meanstd transformation

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])
            from filelock import FileLock
            with FileLock("./data.lock"):
                trainset = torchvision.datasets.CIFAR10(
                    root='~/data',
                    train=True,
                    download=True,
                    transform=transform_train)
            valset = torchvision.datasets.CIFAR10(
                root='~/data',
                train=False,
                download=False,
                transform=transform_test)
            num_classes = 10

            # Create the Dataloaders to feed data to the training and validation steps
            self.train_loader = torch.utils.data.DataLoader(
                trainset,
                batch_size=self.batch_size,
                num_workers=2,
                pin_memory=False)

            self._train_iterator = iter(self.train_loader)
            self.val_loader = torch.utils.data.DataLoader(
                valset,
                batch_size=self.batch_size,
                num_workers=2,
                pin_memory=False,
            )
            self._train_iterator = iter(self.train_loader)

            model_cls = models.__dict__[self.model_string.lower()]
            self.model = model_cls(pretrained=False).cuda()
            logger.info("Finished creating model.")

            # define loss function (criterion) and optimizer
            self.criterion = nn.CrossEntropyLoss().cuda()
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.starting_lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay)
        logger.info("Finished Setup model.")

    def step(self):
        # Set epoch count for DistributedSampler
        logger.debug("Starting step.")
        self.train_sampler.set_epoch(self.epoch)

        # Adjust learning rate according to schedule
        adjust_learning_rate(self.starting_lr, self.optimizer, self.epoch)

        # train for one self.epoch

        logger.debug("Begin Training Epoch {}".format(self.epoch + 1))
        with self._timers["training"]:
            train_stats = train(
                self._train_iterator, self.model, self.criterion,
                self.optimizer, max_steps=self._steps_per_iteration)
            train_stats["epoch"] = self.epoch

        if train_stats.pop("_increment_epoch"):
            self.epoch += 1
            self._train_iterator = iter(self.train_loader)
        else:
            logger.debug("Not incrementing epoch")
        train_stats.update(self.stats())
        return train_stats

    def compute_grads(self):
        try:
            features, target = next(self._train_iterator)
        except StopIteration:
            self.train_iterator = iter(self.train_loader)
            features, target = next(self.train_iterator)

        timers = {k: TimerStat() for k in ["d2h", "fwd", "grad", "apply"]}

        # Create non_blocking tensors for distributed training
        with timers["d2h"]:
            features = features.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # compute output
        with timers["fwd"]:
            output = self.model(features)
            loss = self.criterion(output, target)

        with timers["grad"]:
            # compute gradients in a backward pass
            self.optimizer.zero_grad()
            loss.backward()

        grads = []
        for p in self.model.parameters():
            if p.grad is not None:
                grads.append(p.grad.data.cpu().numpy())
            else:
                grads.append(None)
        return grads

    def merge(self, list_of_grads):
        list_of_grads = ray.get(list_of_grads)
        merged_grads = []
        for i, _ in enumerate(list_of_grads[0]):
            all_models_layer = np.mean(
                [grads[i] for grads in list_of_grads], axis=0)
            merged_grads += [all_models_layer]
        return merged_grads

    def apply_grad(self, gradients):
        for g, p in zip(gradients, self.model.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g).cuda(non_blocking=True)
        self.optimizer.step()

    def stats(self):
        stats = {}
        for k, t in self._timers.items():
            stats[k + "_time_mean"] = t.mean
            stats[k + "_time_total"] = t.sum
            t.reset()
        return stats

    def get_state(self, ckpt_path):
        with self._timers["get_state"]:
            try:
                os.makedirs(ckpt_path)
            except OSError:
                logger.exception("failed making dirs")
            if self.verbose:
                print("getting state")
            state_dict = {}
            tmp_path = os.path.join(ckpt_path,
                                    ".state{}".format(self.world_rank))
            torch.save({
                "model": self.model.state_dict(),
                "opt": self.optimizer.state_dict()
            }, tmp_path)

            with open(tmp_path, "rb") as f:
                state_dict["model_state"] = f.read()

            os.unlink(tmp_path)
            state_dict["epoch"] = self.epoch
            if self.verbose:
                print("Got state.")

        return state_dict

    def set_state(self, state_dict, ckpt_path):
        with self._timers["set_state"]:
            if self.verbose:
                print("setting state for {}".format(self.world_rank))
            try:
                os.makedirs(ckpt_path)
            except OSError:
                print("failed making dirs")
            tmp_path = os.path.join(ckpt_path,
                                    ".state{}".format(self.world_rank))

            with open(tmp_path, "wb") as f:
                f.write(state_dict["model_state"])

            checkpoint = torch.load(tmp_path)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["opt"])

            os.unlink(tmp_path)
            # self.model.train()
            self.epoch = state_dict["epoch"]
            if self.verbose:
                print("Loaded state.")

    def get_host(self):
        return os.uname()[1]

    def node_ip(self):
        return ray.services.get_node_ip_address()

    def get_client(self):
        client_table = ray.global_state.client_table()
        socket_name = str(ray.worker.global_worker.plasma_client.store_socket_name)
        return [k["ClientID"] for k in client_table if k["ObjectStoreSocketName"] == socket_name][0]


class NodeColocatorActor():
    """Object that is called when launching the different nodes

    Should take in N number of gpus in the node (and the location of the cluster?)
    and create N actors with num_gpu=0 and place them on the cluster.
    """

    def __init__(self, batch_size, num_gpus, config):
        RemotePyTorchRunner = ray.remote(PyTorchRunner)
        logger.info(f"Colocator launched on: {os.uname()[1]}")
        args = [
            config["batch_per_device"],
            config["starting_lr"],
            config["momentum"],
            config["weight_decay"],
            config["steps_per_iteration"],
            config["model_string"],
            config["verbose"],
            config["use_nccl"]
        ]
        self.remote_workers = create_colocated(
            RemotePyTorchRunner, args, count=num_gpus)
        gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        logger.info(f"Colocator sharing {gpu_ids}")

        assert len(gpu_ids) == len(self.remote_workers)
        for dev_id, worker in zip(gpu_ids, self.remote_workers):
            worker.set_device.remote(dev_id)

    def get_workers(self):
        return self.remote_workers


class PytorchCustom(ResourceTrainable):
    ADDRESS_TMPL = "tcp://{ip}:{port}"

    def _setup_impl(self, config):
        # get throughput somehow
        self.trial_id = config["trial_id"]
        num_workers = self.resources.extra_gpu
        resources = None
        extra_res = self.resources.get_res_total(self.trial_id)
        if extra_res:
            assert extra_res == num_workers, "Trainable out of sync!"
            resources = {self.trial_id: 1}
        # model_creator = config["model_creator"]
        # devices_per_worker = self.config["devices_per_worker"]
        self.primary_resource = self.resources.extra_gpu
        self._initial_timing = True  # This is set on first restart.
        self.t1 = None  # This is set on first restart
        self._next_iteration_start = time.time()
        self._time_so_far = 0
        self._data_so_far = 0
        self.resource_time = 0

        self.config = config or DEFAULT_CONFIG

        self._placement_set = [1 for i in range(self.primary_resource)]
        if not resources and self.config["placement"] and self.primary_resource > 1:
            self._placement_set = self.config["placement"]

        config["batch_per_device"] = max(
            int(config["target_batch_size"] / self.primary_resource),
            config["min_batch_size"])
        self.batch_per_device = config["batch_per_device"]
        assert sum(self._placement_set) == self.primary_resource
        self.colocators = []
        self.remote_workers = []
        logger.warning(f"Placement set is {self._placement_set}")
        for actors_in_node in self._placement_set:
            if not actors_in_node:
                continue
            if actors_in_node == 1:
                self.remote_workers += [self._create_single_worker(config, resources)]
            else:
                self.remote_workers += self._create_group_workers(actors_in_node, config)

        self._sync_all_workers()

        self.locations = dict(Counter(ray.get(
            [a.get_client.remote() for a in self.remote_workers])))

    def _create_single_worker(self, config, resources):
        print(f"!!!! resources is {resources}")
        RemotePyTorchRunner = ray.remote(num_gpus=1, resources=resources)(PyTorchRunner)
        worker = RemotePyTorchRunner.remote(
            config["batch_per_device"],
            starting_lr=config["starting_lr"],
            momentum=config["momentum"],
            model_string=config["model_string"],
            weight_decay=config["weight_decay"],
            verbose=config["verbose"],
            steps_per_iteration=config["steps_per_iteration"])
        worker.set_device.remote()
        return worker

    def _create_group_workers(self, actors_in_node, config):
        RemoteColocator = ray.remote(num_gpus=int(actors_in_node))(NodeColocatorActor)
        colocator = RemoteColocator.remote(
            self.config["target_batch_size"], int(actors_in_node), self.config)
        self.colocators += [colocator]

        return ray.get(colocator.get_workers.remote())

    def _sync_all_workers(self):
        setup_futures = []
        master_ip = None
        port = int(4000 + random.choice(np.r_[:4000]))

        for world_rank, worker in enumerate(self.remote_workers):
            if not master_ip:
                master_ip = ray.get(worker.node_ip.remote())
            setup_futures += [
                worker.setup_proc_group.remote(
                    self.ADDRESS_TMPL.format(ip=master_ip, port=port),
                    world_rank, len(self.remote_workers))
            ]

        ray.get(setup_futures)
        [worker.setup_model.remote() for worker in self.remote_workers]

        logger.error("TODO FOR REAL WORKLOAD: Get number directory, sync all models")

    def _setup(self, config):
        self.session_timer = {
            "setup": TimerStat(window_size=1),
            "train": TimerStat(window_size=2)
        }
        with self.session_timer["setup"]:
            self._setup_impl(config)

    def train(self):
        random_stop = max(self.resources.extra_gpu * 10, 20)
        with self.session_timer["train"]:
            result = super(PytorchCustom, self).train()

        self.session_timer["train"].push_units_processed(
            result["data_this_epoch"])
        result.update(
            train_throughput=self.session_timer["train"].mean_throughput)
        result["locations"] = self.locations
        print(f"{self._iteration} > {random_stop}")
        result["done"] = self._iteration > random_stop
        return result

    def _train(self):
        NUM_ITER = 10
        for i in range(NUM_ITER):
            all_grads = [w.compute_grads.remote() for w in self.remote_workers]
            merged = self.remote_workers[0].merge.remote(all_grads)
            final = [w.apply_grad.remote(merged) for w in self.remote_workers]
        ray.get(final)
        data_this_epoch = self.batch_per_device * NUM_ITER * len(self.remote_workers)
        self._data_so_far += data_this_epoch
        results = {}
        results.update(data_this_epoch=data_this_epoch)
        results.update(total_data_processed=self._data_so_far)
        if self.config["dataset"] == "CIFAR":
            results.update(epochs_processed=self._data_so_far / 50000)
        results.update(num_workers=len(self.remote_workers))

        self._time_so_far += time.time() - self._next_iteration_start
        self._next_iteration_start = time.time()
        results.update(time_since_start=self._time_so_far)

        return results

    def _save(self, ckpt):
        return {
            "worker_state":
            ray.get(self.remote_workers[0].get_state.remote(ckpt)),
            "time_so_far":
            self._time_so_far,
            "data_so_far":
            self._data_so_far,
            "resource_time": self.resource_time,
            "ckpt_path":
            ckpt,
            "t1":
            self.t1 or self.session_timer["train"].mean,
        }

    def _restore(self, ckpt):
        self._time_so_far = ckpt["time_so_far"]
        self._data_so_far = ckpt["data_so_far"]
        self.t1 = ckpt["t1"]
        self.resource_time = ckpt["resource_time"]
        self._initial_timing = False
        worker_state = ray.put(ckpt["worker_state"])
        states = []

        for worker in self.remote_workers:
            states += [
                worker.set_state.remote(worker_state, ckpt["ckpt_path"])
            ]

        ray.get(states)
        self._next_iteration_start = time.time()

    def _stop(self):
        logger.warning("Calling stop on this trainable.")
        stops = []
        for colocator in self.colocators:
            stops += [colocator.__ray_terminate__.remote()]

        for worker in self.remote_workers:
            stops += [worker.__ray_terminate__.remote()]

    def ready_to_resize(self):
        """Determines whether the overhead has been accounted for.

        N > frac{C}{((t_1)* (1 - 1/R) - c_1)}

        C is starting overhead. N is number of iterations.
        """
        if not self.t1:  # This allows resizing whenever the scheduler wants.
            return True

        train_timer = self.session_timer["train"]

        if train_timer.size < 3:
            return False
        diff = self.t1 - train_timer.median
        overhead = self.session_timer["setup"].first + (
            train_timer.sum - (train_timer.median * train_timer.size))
        logger.warning("For {}: Overhead: {:0.4f}; Iterations needed: {}".format(
            self, overhead, int(overhead / diff)))
        return train_timer.size > overhead / diff


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    runner = PyTorchRunner(128)
    runner.set_device()
    runner.setup_proc_group()
    runner.setup_model()
    import ipdb; ipdb.set_trace()
    grads = runner.compute_grads()
    runner.merge([grads, grads])
    runner.apply_grad(grads)


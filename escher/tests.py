from escher.distributed_trainable import ResourceExecutor, Aggregator
import sys
import os
from escher.pytorch_trainable import PytorchSGD, DEFAULT_CONFIG

from escher.placement import PlacementScheduler

import ray
import time
from ray import tune
from datetime import datetime
from ray.tests.cluster_utils import Cluster
from ray.tune.trial import Resources
import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter, description="Shard.")

parser.add_argument(
    "--redis-address",
    default=None,
    type=str,
    help="The Redis address of the cluster.")
args = parser.parse_args(sys.argv[1:])

def test_pytorch_custom():
    
    ray.init(redis_address=args.redis_address)
    from escher.pytorch_custom import PytorchCustom, DEFAULT_CONFIG
    config = DEFAULT_CONFIG.copy()
    config["trial_id"] = "hi"
    config["min_batch_size"] = 4
    config["target_batch_size"] = 4
    config["model_string"] = "resnet101"
    config["placement"] = [3, 2]
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    custom = PytorchCustom(config, resources=Resources(0,0, extra_gpu=5))
    for i in range(10):
        print(custom.train())
    

def test_basic():
    from collections import Counter
    # This example demonstrates co-location by creating a resource where the first task lands.
    # NUM_NODES = 3

    # Initialize cluster
    print("Initializing cluster..")
    cluster = Cluster()
    MOCK = "trialme"
    cluster.add_node(num_cpus=4, resources={MOCK: 2})
    cluster.add_node(num_cpus=4, resources={MOCK: 1})
    cluster.add_node(num_cpus=4, resources={MOCK: 1})

    cluster.wait_for_nodes()

    print("Cluster init complete, connecting driver")
    ray.init(redis_address=cluster.redis_address)



    actor = Aggregator(config={"trial_id": MOCK}, resources=Resources(
        0, 0, extra_custom_resources={MOCK: 4}))
    result = actor.train()
    import ipdb; ipdb.set_trace()


def test_tune_local():
    # Initialize cluster
    print("Initializing cluster..")
    cluster = Cluster()
    cluster.add_node(num_cpus=4)
    cluster.add_node(num_cpus=4)

    cluster.wait_for_nodes()
    LOCAL_MODE = False

    print("Cluster init complete, connecting driver")
    ray.init(redis_address=cluster.redis_address, local_mode=LOCAL_MODE)

    config = {
        "config": {"stop": 20}
    }
    scheduler = PlacementScheduler(4)
    tune.run(Aggregator,
        name="my_exp_2",
        scheduler=scheduler,
        resources_per_trial=tune.grid_search([
            dict(cpu=0, gpu=0, extra_cpu=i)
            for i in [1, 1, 1, 1, 4]]),
        trial_executor=ResourceExecutor(),
        **config)

def timestring():
    return datetime.today().strftime("%Y-%m-%d_%H-%M-%S")

def test_tune():
    ray.init(redis_address="localhost:6379")
    d_config = DEFAULT_CONFIG.copy()
    d_config.update({"target_batch_size": 8, "min_batch_size": 8, "stop": 2, "steps_per_iteration": 1})
    config = {"config": d_config, "stop": {"time_total_s": 90}} 

    scheduler = PlacementScheduler(8)
    tune.run(PytorchSGD,
        name="my_exp_{}".format(timestring()),
        local_dir="~/results/",
        scheduler=scheduler,
        resources_per_trial=tune.grid_search([
            dict(cpu=0, gpu=0, extra_gpu=i)
            for i in [1,1,1,1, 6]]),
        trial_executor=ResourceExecutor(),
        **config)

    tune.run(PytorchSGD,
        name="no_sched_{}".format(timestring()),
        local_dir="~/results/",
        resources_per_trial=tune.grid_search([
            dict(cpu=0, gpu=0, extra_gpu=i)
            for i in [1,1,1,1, 6]]),
        trial_executor=ResourceExecutor(),
        **config)


if __name__ == '__main__':
    test_pytorch_custom()

from escher.distributed_trainable import ResourceExecutor, Aggregator
from escher.placement import PlacementScheduler

import ray
import time
from ray import tune
from ray.tests.cluster_utils import Cluster
from ray.tune.trial import Resources

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


def test_tune():
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
        name="my_exp",
        scheduler=scheduler,
        resources_per_trial=tune.grid_search([
            dict(cpu=0, gpu=0, extra_cpu=i)
            for i in [1, 1, 1, 1, 4]]),
        trial_executor=ResourceExecutor(),
        **config)


if __name__ == '__main__':
    test_tune()

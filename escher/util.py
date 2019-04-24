from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import filelock
import json
import logging
import numpy as np
import os
import time

import ray

logger = logging.getLogger(__name__)


class TimerStat(object):
    """A running stat for conveniently logging the duration of a code block.

    Example:
        wait_timer = TimerStat()
        with wait_timer:
            ray.wait(...)

    Note that this class is *not* thread-safe.
    """

    def __init__(self, window_size=10):
        self._window_size = window_size
        self._samples = []
        self._units_processed = []
        self._start_time = None
        self._total_time = 0.0
        self.count = 0

    def __enter__(self):
        assert self._start_time is None, "concurrent updates not supported"
        self._start_time = time.time()

    def __exit__(self, type, value, tb):
        assert self._start_time is not None
        time_delta = time.time() - self._start_time
        self.push(time_delta)
        self._start_time = None

    def push(self, time_delta):
        self._samples.append(time_delta)
        if len(self._samples) > self._window_size:
            self._samples.pop(0)
        self.count += 1
        self._total_time += time_delta

    def push_units_processed(self, n):
        self._units_processed.append(n)
        if len(self._units_processed) > self._window_size:
            self._units_processed.pop(0)

    @property
    def mean(self):
        return np.mean(self._samples)

    @property
    def median(self):
        return np.median(self._samples)

    @property
    def sum(self):
        return np.sum(self._samples)

    @property
    def max(self):
        return np.max(self._samples)

    @property
    def first(self):
        return self._samples[0] if self._samples else None

    @property
    def last(self):
        return self._samples[-1] if self._samples else None

    @property
    def size(self):
        return len(self._samples)

    @property
    def mean_units_processed(self):
        return float(np.mean(self._units_processed))

    @property
    def mean_throughput(self):
        time_total = sum(self._samples)
        if not time_total:
            return 0.0
        return sum(self._units_processed) / time_total

    def reset(self):
        self._samples = []
        self._units_processed = []
        self._start_time = None
        self._total_time = 0.0
        self.count = 0

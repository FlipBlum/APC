# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2023 Comet ML INC
#  This file can not be copied and/or distributed without
#  the express permission of Comet ML Inc.
# *******************************************************
import abc
import logging
import threading
import time

from comet_ml.system.system_metrics_types import (
    CompatibleSystemMetrics,
    SystemMetricsCallable,
)

TOLERATE_NUM_FAILURES = 10
MINIMUM_INTERVAL = 10

LOGGER = logging.getLogger(__name__)


class BaseSystemLoggingThread(threading.Thread):
    """
    The base class for all system metrics logger threads implementing common functionality.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(
        self,
        initial_interval: int,
        callback: SystemMetricsCallable,
        logger: logging.Logger,
    ) -> None:
        threading.Thread.__init__(self, daemon=True)
        self.interval = (
            initial_interval
            if self._interval_is_valid(initial_interval)
            else MINIMUM_INTERVAL
        )
        self.callback = callback
        self.logger = logger

        self.last_run = 0.0
        self.closed = False

    def run(self):
        failures = 0
        while not self.closed:
            try:
                self._loop()
            except Exception:
                self.logger.debug(
                    "System Metric Thread unexpectedly failed to run", exc_info=True
                )
                time.sleep(1)
                failures += 1
                if failures > TOLERATE_NUM_FAILURES:
                    break

    def _loop(self):
        if self._should_run():
            # Run
            metrics = self.get_metrics()
            if _not_empty(metrics) > 0:
                self.callback(metrics)
            self.last_run = time.time()

        time.sleep(1)

    def _should_run(self) -> bool:
        next_run = self.last_run + self.interval  # seconds
        now = time.time()
        result = next_run <= now
        return result

    def _interval_is_valid(self, interval: int):
        if interval >= MINIMUM_INTERVAL:
            return True

        LOGGER.warning(
            "Provided interval is too low, falling back to the minimum interval (10 Seconds)"
        )
        return False

    def close(self):
        self.closed = True

    @abc.abstractmethod
    def get_metrics(self) -> CompatibleSystemMetrics:
        pass


def _not_empty(metrics):
    if metrics is None:
        return False

    if hasattr(metrics, "__len__") or hasattr(metrics, "len"):
        return len(metrics) > 0

    return True

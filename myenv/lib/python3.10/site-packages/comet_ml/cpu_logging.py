# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at https://www.comet.com
#  Copyright (C) 2015-2021 Comet ML INC
#  This file can not be copied and/or distributed without
#  the express permission of Comet ML Inc.
# *******************************************************

"""
Author: Douglas Blank

This module contains the main components of CPU information logging

"""
import logging

from comet_ml.system.base_system_logging_thread import BaseSystemLoggingThread
from comet_ml.system.cpu import utilization
from comet_ml.system.system_metrics_types import (
    NamedSystemMetrics,
    SystemMetricsCallable,
)

try:
    import psutil
except Exception:
    psutil = None

LOGGER = logging.getLogger(__name__)


def is_cpu_info_available():
    return psutil is not None


class CPULoggingThread(BaseSystemLoggingThread):
    def __init__(
        self,
        initial_interval: int,
        callback: SystemMetricsCallable,
        include_cpu_per_core: bool,
        include_compute_metrics: bool,
    ):
        super(CPULoggingThread, self).__init__(
            initial_interval=initial_interval,
            callback=callback,
            logger=LOGGER,
        )
        self.include_compute_metrics = include_compute_metrics
        self._include_cpu_per_core = include_cpu_per_core
        self.name = "CPULoggingThread"

        LOGGER.debug(
            "CPUThread create with %ds interval",
            initial_interval,
        )

    def get_metrics(self) -> NamedSystemMetrics:
        vm = psutil.virtual_memory()
        metrics = {}
        percents = psutil.cpu_percent(interval=None, percpu=True)
        # CPU percents:
        if len(percents) > 0:
            avg_percent = sum(percents) / len(percents)
            metrics["sys.cpu.percent.avg"] = avg_percent

            if self.include_compute_metrics:
                metrics["sys.compute.overall"] = round(avg_percent, 1)
                metrics["sys.compute.utilized"] = utilization.process_tree()

            if self._include_cpu_per_core:
                for (i, percent) in enumerate(percents):
                    metrics["sys.cpu.percent.%02d" % (i + 1)] = percent
        # Load average:
        try:
            # psutil <= 5.6.2 did not have getloadavg:
            if hasattr(psutil, "getloadavg"):
                metrics["sys.load.avg"] = psutil.getloadavg()[0]
            else:
                # Do not log an empty metric
                pass
        except OSError:
            metrics["sys.load.avg"] = None

        # RAM:
        metrics["sys.ram.total"] = vm.total
        metrics["sys.ram.used"] = vm.total - vm.available
        return metrics

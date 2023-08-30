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
import logging
from typing import Any, Callable, Dict

from comet_ml.system.network.network_rates import NetworkRatesMonitor

try:
    import psutil
except Exception:
    psutil = None

from comet_ml.system.base_system_logging_thread import BaseSystemLoggingThread

LOGGER = logging.getLogger(__name__)


def is_network_info_available():
    return psutil is not None


class NetworkLoggingThread(BaseSystemLoggingThread):
    def __init__(
        self,
        initial_interval: int,
        callback: Callable[[Dict[str, Any]], None],
    ):
        super(NetworkLoggingThread, self).__init__(
            initial_interval=initial_interval,
            callback=callback,
            logger=LOGGER,
        )

        self.network_monitor = NetworkRatesMonitor()
        self.name = "NetworkLoggingThread"

        LOGGER.debug(
            "NetworkLoggingThread created with %ds interval",
            initial_interval,
        )

    def get_metrics(self) -> Dict[str, Any]:
        result = self.network_monitor.current_rate()
        if result is None:
            return {}
        metrics = {
            "sys.network.send_bps": result.bytes_sent_rate,
            "sys.network.receive_bps": result.bytes_recv_rate,
        }
        return metrics

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
#  This file can not be copied and/or distributed
#  without the express permission of Comet ML Inc.
# *******************************************************
import multiprocessing
from ctypes import c_int


class FileUploadLimitsGuard:
    def __init__(self):
        self._image_upload_limit_reached = multiprocessing.Value(c_int, 0)

    def image_upload_limit_exceeded(self):
        self._image_upload_limit_reached.value = 1

    def has_image_upload_limit_exceeded(self) -> bool:
        return self._image_upload_limit_reached.value == 1

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.1.0
# License: MIT
# See the documentation at gyptis.gitlab.io
"""
Information about the package.
"""

import importlib.metadata as metadata

try:
    data = metadata.metadata("gyptis")
    __version__ = metadata.version("gyptis")
    __author__ = data.get("author")
    __description__ = data.get("summary")
except Exception:
    __version__ = "unknown"
    __author__ = "unknown"
    __description__ = "unknown"

__name__ = "gyptis"

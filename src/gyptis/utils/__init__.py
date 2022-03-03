#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io


from IPython import get_ipython

from .helpers import *
from .jupyter import VersionTable
from .log import *
from .parallel import *
from .sample import *
from .time import *

_IP = get_ipython()
if _IP is not None:
    _IP.register_magics(VersionTable)

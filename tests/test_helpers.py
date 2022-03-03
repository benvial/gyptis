#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io

import numpy as np

from gyptis.utils.helpers import *


def test_all():
    assert tanh(1.2) == np.tanh(1.2)

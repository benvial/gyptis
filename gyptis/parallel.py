#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

from functools import wraps

import numpy as np
from joblib import Parallel, delayed


def parloop(n_jobs=1):
    def deco_parloop(func):
        """
        Decorate a function to parallelize.
        """

        @wraps(func)
        def my_func(*args, **kwargs):
            other_args = args[1:]
            return Parallel(n_jobs=n_jobs)(
                delayed(func)(x, *other_args, **kwargs) for x in args[0]
            )

        return my_func

    return deco_parloop

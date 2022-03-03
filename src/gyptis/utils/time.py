#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io


__all__ = ["tic", "toc", "list_time"]


import time

from .. import dolfin


def list_time():
    return dolfin.list_timings(dolfin.TimingClear.clear, [dolfin.TimingType.wall])


def tic():
    return time.time()


def toc(t0, verbose=True):
    t = time.time() - t0
    if verbose:
        print(f"elapsed time {t}s")
    return t

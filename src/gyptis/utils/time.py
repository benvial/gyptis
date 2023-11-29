#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io


__all__ = ["tic", "toc", "list_time"]


import time

from .. import dolfin


def list_time():
    cols = [dolfin.TimingType.wall, dolfin.TimingType.system, dolfin.TimingType.user]
    return dolfin.list_timings(dolfin.TimingClear.clear, cols)


def tic():
    return time.time()


def toc(t0, verbose=True):
    t = time.time() - t0
    if verbose:
        print(f"elapsed time {t}s")
    return t

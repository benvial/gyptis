#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

__all__ = ["format_matrix", "logger", "set_log_level"]


import logging
import sys

from loguru import logger

from .. import dolfin


def set_log_level(level):
    global logger
    logger.remove()
    format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> -- <level>{message}</level>"
    logger.add(sys.stderr, format=format, level=level, colorize=True)


def format_matrix(m, ndigit=4, extra_space=0, cplx=False):
    def printim(y):
        return f"{sgn(float(y))}{abs(float(y))}j"

    def sgn(u):
        if np.sign(u) == -1:
            return "-"
        else:
            return "+"

    dim = len(m[0])

    pad = " " * extra_space

    if cplx:
        m = [[_.real, _.imag] for _ in np.ravel(m)]

    a = [f"%.{ndigit}f" % elem for elem in np.ravel(m)]

    if dim == 3:
        if cplx:
            b = f"""[{a[0]}{printim(a[1])} {a[2]}{printim(a[3])}
            {a[4]}{printim(a[5])}] \n{pad}[{a[6]}{printim(a[7])}
            {a[8]}{printim(a[9])} {a[10]}{printim(a[11])}]
            \n{pad}[{a[12]}{printim(a[13])} {a[14]}{printim(a[15])}
            {a[16]}{printim(a[17])}]
            """

        else:
            b = f"""[{a[0]} {a[1]} {a[2]}] \n{pad}[{a[3]} {a[4]} {a[5]}]
             \n{pad}[{a[6]} {a[7]} {a[8]}]
             """
    else:
        if cplx:
            b = f"""[{a[0]}{printim(a[1])} {a[2]}{printim(a[3])}]
            \n{pad}[{a[4]}{printim(a[5])} {a[6]}{printim(a[7])}]
            """
        else:
            b = f"[{a[0]} {a[1]}] \n{pad}[{a[2]} {a[3]}]"
    return b


set_log_level(logging.INFO)
dolfin.set_log_level(logging.INFO)

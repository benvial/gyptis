#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.1.2
# License: MIT
# See the documentation at gyptis.gitlab.io


import os

meshpar = 1

for i in range(1, 6):
    print("======================================")
    print(f"============  {i} proc  ================")
    print("======================================")
    # os.system(f"time mpirun -n {i} python parallel_int.py {meshpar}")
    os.system(f"time mpirun -n {i} python min_parallel_int.py {meshpar}")

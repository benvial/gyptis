#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


import os

meshpar = 1

for i in range(1, 6):
    print("======================================")
    print(f"============  {i} proc  ================")
    print("======================================")
    # os.system(f"time mpirun -n {i} python parallel_int.py {meshpar}")
    os.system(f"time mpirun -n {i} python min_parallel_int.py {meshpar}")

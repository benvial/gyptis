#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import pytest
from gyptis.mesh import *


def test_marked_mesh(shared_datadir):
    mshfile  = (shared_datadir / 'mesh.msh')
    mmsh = MarkedMesh(mshfile, 2)
    print(mmsh)

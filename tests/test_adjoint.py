#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io


def test_adjoint():
    import os

    initadj = os.environ.get("GYPTIS_ADJOINT") is not None

    import gyptis

    gyptis.use_adjoint(True)
    assert gyptis.ADJOINT is True
    assert gyptis.dolfin.__name__ == "dolfin_adjoint"

    gyptis.use_adjoint(False)
    assert gyptis.ADJOINT is False
    assert gyptis.dolfin.__name__ == "dolfin"

    if initadj:
        gyptis.use_adjoint(True)

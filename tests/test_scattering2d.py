#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

import os

import numpy as np
import pytest

import gyptis
from gyptis import BoxPML, Scattering
from gyptis.plot import *
from gyptis.scattering2d import *
from gyptis.utils.helpers import list_time

plt.ion()

polarization = "TM"
degree = 2
wavelength = 0.3
pmesh = 3
lmin = wavelength / pmesh

geom = BoxPML(
    dim=2, box_size=(4 * wavelength, 4 * wavelength), pml_width=(wavelength, wavelength)
)
cyl = geom.add_circle(0, 0, 0, 0.2)
cyl, box = geom.fragment(cyl, geom.box)
geom.add_physical(box, "box")
geom.add_physical(cyl, "cyl")
[geom.set_size(pml, lmin) for pml in geom.pmls]
geom.set_size("box", lmin)
geom.set_size("cyl", lmin)
geom.build()
mesh = geom.mesh_object["mesh"]
markers = geom.mesh_object["markers"]["triangle"]
mapping = geom.subdomains["surfaces"]

epsilon = dict(box=1, cyl=3)
mu = dict(box=1, cyl=1)


@pytest.mark.parametrize(
    "degree,polarization", [(1, "TM"), (2, "TM"), (1, "TE"), (2, "TE")]
)
def test_scatt2d_pw(degree, polarization):

    pw = PlaneWave(wavelength=wavelength, angle=0, dim=2, domain=mesh, degree=degree)

    s = Scattering(geom, epsilon, mu, pw, degree=degree, polarization=polarization)
    u = s.solve()
    list_time()
    print(assemble(u * s.formulation.dx))
    plt.ion()
    if gyptis.ADJOINT:
        eps_max, eps_min = 3, 1
        Actrl = dolfin.FunctionSpace(mesh, "DG", 0)
        ctrl0 = dolfin.Expression("0.1", degree=2)
        ctrl = project(ctrl0, Actrl)
        eps_lens_func = ctrl * (eps_max - eps_min) + eps_min
        dolfin.set_working_tape(dolfin.Tape())
        h = dolfin.Function(Actrl)
        h.vector()[:] = 1e-2 * np.random.rand(Actrl.dim())
        # epsilon["cyl"] = project(eps_lens_func,s.formulation.real_function_space)
        epsilon["cyl"] = eps_lens_func
        # project(eps_lens_func,s.formulation.real_function_space)
        s = Scatt2D(geom, epsilon, mu, pw, degree=degree, polarization=polarization)
        field = s.solve()
        J = -assemble(inner(field, field.conj) * s.dx("box")).real
        Jhat = dolfin.ReducedFunctional(J, dolfin.Control(ctrl))
        conv_rate = dolfin.taylor_test(Jhat, ctrl, h)
        print("convergence rate = ", conv_rate)
        assert abs(conv_rate - 2) < 1e-2


@pytest.mark.parametrize(
    "degree,polarization", [(1, "TM"), (2, "TM"), (1, "TE"), (2, "TE")]
)
def test_scatt2d_ls(degree, polarization):
    gf = LineSource(wavelength, (-wavelength, 0), domain=mesh, degree=degree)
    s = Scatt2D(geom, epsilon, mu, gf, degree=degree, polarization=polarization)
    u = s.solve()
    list_time()
    tot = gf.expression + u
    s.source.position = (0, -wavelength)
    s.assemble_rhs()
    u1 = s.solve_system(again=True)
    list_time()

    s.plot_field()
    s.animate_field(n=2, filename="animation.gif")
    os.system("rm -rf animation.gif")


@pytest.mark.parametrize("polarization", ["TM", "TE"])
def test_scatt2d_pec(polarization):

    geom = BoxPML2D(
        box_size=(4 * wavelength, 4 * wavelength),
        pml_width=(wavelength, wavelength),
    )
    cyl = geom.add_circle(0, 0, 0, 0.2)
    box = geom.cut(geom.box, cyl)
    geom.add_physical(box, "box")
    cyl_bnds = geom.get_boundaries("box")[-1]
    geom.add_physical(cyl_bnds, "cyl_bnds", dim=1)
    [geom.set_size(pml, lmin) for pml in geom.pmls]
    geom.set_size("box", lmin)
    geom.build(0)
    mesh = geom.mesh_object["mesh"]
    markers = geom.mesh_object["markers"]["triangle"]
    mapping = geom.subdomains["surfaces"]
    epsilon = dict(box=1)
    mu = dict(box=1)

    pw = PlaneWave(wavelength=wavelength, angle=0, dim=2, domain=mesh, degree=degree)

    bcs = {"cyl_bnds": "PEC"}
    s = Scatt2D(
        geom,
        epsilon,
        mu,
        pw,
        degree=degree,
        polarization=polarization,
        boundary_conditions=bcs,
    )

    u = s.solve()
    list_time()
    print(assemble(u * s.formulation.dx))


@pytest.mark.parametrize("polarization", ["TM", "TE"])
def test_scatt2d_scs(polarization):
    pmesh = 6
    wavelength = 452
    eps_core = 2
    eps_shell = 6

    R1 = 60
    R2 = 30
    Rcalc = 2 * R1
    lmin = wavelength / pmesh
    pml_width = wavelength

    lbox = Rcalc * 2 * 1.1
    geom = BoxPML(
        dim=2,
        box_size=(lbox, lbox),
        pml_width=(pml_width, pml_width),
        Rcalc=Rcalc,
    )
    box = geom.box
    shell = geom.add_circle(0, 0, 0, R1)
    out = geom.fragment(shell, box)
    box = out[1:3]
    shell = out[0]
    core = geom.add_circle(0, 0, 0, R2)
    core, shell = geom.fragment(core, shell)
    geom.add_physical(box, "box")
    geom.add_physical(core, "core")
    geom.add_physical(shell, "shell")
    [geom.set_size(pml, lmin * 0.7) for pml in geom.pmls]
    geom.set_size("box", lmin)
    geom.set_size("core", lmin / eps_core ** 0.5)
    geom.set_size("shell", lmin / eps_shell ** 0.5)
    geom.build()
    pw = PlaneWave(wavelength=wavelength, angle=0, dim=2, domain=geom.mesh, degree=2)
    epsilon = dict(box=1, core=eps_core, shell=eps_shell)
    mu = dict(box=1, core=1, shell=1)

    s = Scattering(
        geom,
        epsilon,
        mu,
        pw,
        degree=2,
        polarization="TE",
    )
    s.solve()
    cs = s.get_cross_sections()
    assert np.allclose(
        cs["extinction"], cs["scattering"] + cs["absorption"], rtol=1e-12
    )
    print(cs["extinction"])
    print(cs["scattering"] + cs["absorption"])
    print(abs(cs["scattering"] + cs["absorption"] - cs["extinction"]))

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io


import numpy as np
import pytest


def test_simple(tol=1e-12):
    from gyptis.complex import Complex, Constant

    x, y, x1, y1 = np.random.rand(4) - 0.5
    z = Complex(x, y)
    z1 = Complex(x1, y1)
    q = x + 1j * y
    q1 = x1 + 1j * y1
    assert str(z) == f"({x} + {y}j)"
    assert repr(z) == f"Complex({x}, {y})"
    assert z == q
    assert -z == Complex(-x, -y)
    assert z != Complex(x / 2, y)
    assert z != Complex(x, y / 2)
    assert z != Complex(x / 2, y / 2)
    assert z != Complex(x, y) / 2
    assert z != q / q1
    assert z != q / 2
    assert 2 * z == Complex(2 * x, 2 * y)
    assert z / z == 1
    assert 1 / z == 1 / z
    assert z * z1 == q * q1
    assert z * q1 == q * z1
    assert abs(z / q1 - q / z1) < tol
    assert abs(z / z1 - q / q1) < tol
    assert abs(z) == (x**2 + y**2) ** 0.5
    assert z.conj == q.conj()
    assert z.module == abs(z)
    assert z.phase == np.angle(q)
    assert abs(z**2 - q**2) < tol
    assert abs(z ** (-3.4) - q ** (-3.4)) < tol
    assert z + q1 == z1 + q
    assert len(Complex(1, 1)) == 0
    assert len(Complex([1, 2], [1, 2])) == 2
    assert Complex(np.random.rand(3, 2), np.random.rand(3, 2)).shape == (3, 2)

    with pytest.raises(ValueError):
        len(Complex([1], [3, 2]))

    with pytest.raises(NotImplementedError):
        z**z

    Z = Complex([1, 2], [3, 2])
    for i, r in enumerate(Z):
        assert r == Z[i]

    Constant((1 + 6 * 1j, 6 - 0.23j), name="test")
    Constant((1 + 6 * 1j, 6 - 0.23j))
    Constant(3 + 1j)
    Constant(12)
    Constant((1, 2))


def test_complex(tol=1e-15):
    import gyptis
    from gyptis import dolfin
    from gyptis.bc import DirichletBC

    nx, ny = 50, 50
    mesh = dolfin.UnitSquareMesh(nx, ny)

    def boundary(x):
        return (
            x[0] < dolfin.DOLFIN_EPS
            or x[0] > 1.0 - dolfin.DOLFIN_EPS
            or x[1] < dolfin.DOLFIN_EPS
            or x[1] > 1.0 - dolfin.DOLFIN_EPS
        )

    W = gyptis.ComplexFunctionSpace(mesh, "CG", 1)
    # W0 =  dolfin.FunctionSpace(mesh,W.ufl_element().extract_component(0)[1])
    W0 = dolfin.FunctionSpace(mesh, "DG", 0)

    # u = Function(W)
    utrial = gyptis.TrialFunction(W)
    utest = gyptis.TestFunction(W)
    dx = dolfin.dx(domain=mesh)

    k = 2
    qre, qim = 2, -1
    gamma = 0.05
    expr = f"exp(-pow((x[0]-0.5)/{gamma},2)-pow((x[1]-0.5)/{gamma},2))"
    expr_re = f"{qre}+{expr}"
    expr_im = f"{qim}+{expr}"
    sol = dolfin.Expression((expr_re, expr_im), degree=1, domain=mesh)
    sol = gyptis.Complex(sol[0], sol[1])

    source = (k**2) * sol * utest + gyptis.inner(gyptis.grad(sol), gyptis.grad(utest))

    F = (
        gyptis.inner(gyptis.grad(utrial), gyptis.grad(utest)) * dx
        + k**2 * utrial * utest * dx
        - source * dx
    )
    F = F.real + F.imag

    u_dirichlet = qre + 1j * qim
    bcre, bcim = DirichletBC(W, u_dirichlet, boundary)

    bcs = [bcre, bcim]
    Lh = dolfin.rhs(F)
    bh = gyptis.assemble(Lh)
    ah = dolfin.lhs(F)
    Ah = gyptis.assemble(ah)
    dolfin.VectorFunctionSpace(mesh, "CG", 2)

    # u = Function(W)

    u = dolfin.Function(W)

    def _solve(solver, u, tol, tol_local):
        for bc in bcs:
            bc.apply(Ah, bh)
        solver.solve(Ah, u.vector(), bh)

        u = gyptis.Complex(u[0], u[1])
        assert abs(gyptis.assemble((u - sol) * dx)) < tol
        assert abs((u((0.5, 0.5)) - sol((0.5, 0.5)))) < tol_local
        return u

    # _solve(dolfin.LUSolver(Ah), 1e-12, 1e-12)
    u = _solve(dolfin.PETScKrylovSolver(), u, 1e-6, 1e-4)

    gyptis.project(u.real, W0)
    gyptis.inner(u.real, u)
    gyptis.inner(u, u.imag)
    gyptis.inner(u.real, u.imag)

    u.phase
    u.module

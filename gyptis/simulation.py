#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT

from . import dolfin
from .complex import Complex, assemble


class Simulation:
    def __init__(self, geometry, formulation=None):
        self.geometry = geometry
        self.formulation = formulation
        self.coefficients = formulation.coefficients
        self.function_space = formulation.function_space
        self._source = formulation.source
        self.boundary_conditions = formulation.boundary_conditions
        self.mesh = self.geometry.mesh
        self._boundary_conditions = []
        self.dx = formulation.dx
        self.ds = formulation.ds
        self.dS = formulation.dS

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, value):
        self._source = value

    def assemble_lhs(self, **kwargs):
        self.formulation.build_lhs(**kwargs)
        self.matrix = assemble(self.formulation.lhs)
        return self.matrix

    def assemble_rhs(self, **kwargs):
        self.formulation.build_rhs(**kwargs)
        self.vector = assemble(self.formulation.rhs)
        return self.vector

    def assemble(self, **kwargs):
        self.matrix = self.assemble_lhs(**kwargs)
        self.vector = self.assemble_rhs(**kwargs)
        return self.matrix, self.vector

    def apply_boundary_conditions(self):
        bcs = self.formulation.build_boundary_conditions()
        for bc in bcs:
            bc.apply(self.matrix, self.vector)

    def solve_system(self, again=False, vector_function=True):
        if vector_function:
            element = self.function_space.split()[0].ufl_element()
            V_vect = dolfin.VectorFunctionSpace(
                self.mesh, element.family(), element.degree()
            )
            u = dolfin.Function(V_vect)
        else:
            u = dolfin.Function(self.function_space)

        if not again:
            self.solver = dolfin.LUSolver(self.matrix, "mumps")

        self.solver.solve(u.vector(), self.vector)
        solution = Complex(*u.split())
        return solution

    def solve(self, again=False, **kwargs):
        self.assemble(**kwargs)
        self.apply_boundary_conditions()
        return self.solve_system(again=again, **kwargs)

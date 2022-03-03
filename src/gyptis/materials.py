#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io


import copy
import os

import numpy as np

from .complex import *
from .plot import plot


class PML:
    def __init__(
        self,
        direction="x",
        stretch=1 - 1j,
        matched_domain=None,
        applied_domain=None,
    ):
        self.direction = direction
        self.stretch = stretch
        self.matched_domain = matched_domain
        self.applied_domain = applied_domain

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, value):
        valid = ["x", "y", "z", "xy", "xz", "yz", "xyz"]
        if value not in valid:
            raise ValueError(f"Unkown direction {value}, must be one of {valid}")

        self._direction = value

    def jacobian_matrix(self):
        if self.direction == "x":
            s = self.stretch, 1, 1
        elif self.direction == "y":
            s = 1, self.stretch, 1
        elif self.direction == "z":
            s = 1, 1, self.stretch
        elif self.direction == "xy":
            s = self.stretch, self.stretch, 1
        elif self.direction == "xz":
            s = self.stretch, 1, self.stretch
        elif self.direction == "yz":
            s = 1, self.stretch, self.stretch
        else:
            s = self.stretch, self.stretch, self.stretch
        return np.diag(s)

    def transformation_matrix(self):
        J = self.jacobian_matrix()
        invJ = np.linalg.inv(J)
        return invJ @ invJ.T * np.linalg.det(J)


class _SubdomainPy(dolfin.UserExpression):
    def __init__(self, markers, subdomains, mapping, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.markers = markers
        self.subdomains = subdomains
        self.mapping = mapping

    def eval_cell(self, values, x, cell):
        for sub, val in self.mapping.items():
            if self.markers[cell.index] == self.subdomains[sub]:
                if callable(val):
                    values[:] = val(x)
                else:
                    values[:] = val

    def value_shape(self):
        return ()


class _SubdomainCpp(dolfin.CompiledExpression):
    def __init__(self, markers, subdomains, mapping, **kwargs):

        here = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(here, "subdomain.cpp")) as f:
            subdomain_code = f.read()
        compiled_cpp = dolfin.compile_cpp_code(subdomain_code).SubdomainCpp()
        super().__init__(
            compiled_cpp,
            markers=markers,
            subdomains=subdomains,
            mapping=mapping,
            **kwargs,
        )
        self.markers = markers
        self.subdomains = subdomains
        self.mapping = mapping


def _flatten_list(k):
    result = list()
    for i in k:
        if isinstance(i, list):
            result.extend(_flatten_list(i))  # Recursive call
        else:
            result.append(i)
    return result


def _separate_mapping_parts(mapping):
    map_re = {}
    map_im = {}
    for k, v in mapping.items():
        try:
            vre = v.real
            vim = v.imag
        except:
            vre = v
            vim = 0 * v
        map_re[k] = vre
        map_im[k] = vim
    return map_re, map_im


def _apply(item, fun):
    if isinstance(item, list):
        return [_apply(x, fun) for x in item]
    else:
        return fun(item)


def isiter(v):
    return hasattr(v, "__contains__")
    # test = v.real if iscomplex(v) else v
    # if hasattr(test, "ufl_shape"):
    #     return test.ufl_shape != ()
    # else:
    #     return hasattr(test, "__contains__")


def _make_tensor(mapping):
    mapping_tens = mapping.copy()
    N = max([len(v) for v in mapping.values() if isiter(v)])
    id = np.eye(N).tolist()
    for k, v in mapping.items():
        if not isiter(v):

            def fun(id):
                if id == 1:
                    id = v
                return id

            mapping_tens[k] = _apply(id, fun)
        else:
            if type(v) == np.ndarray:
                mapping_tens[k] = v.tolist()
    return mapping_tens


def _fldict(k, vflat):
    lnew = []
    for val in vflat:
        lnew.append({k: val})
    return lnew


def _dic2list(dic):
    k = dic.keys()
    val = dic.values()
    # N = np.shape(list(val)[0])
    N = np.array(list(val)[0]).shape
    nb_keys = len(k)
    dnew = {}
    L = []
    for k, v in dic.items():
        vflat = _flatten_list(v)
        L.append(_fldict(k, vflat))
    L = np.reshape(L, (nb_keys, np.product(N))).T
    o = []
    for l in L:
        d = dict(l[0])
        for a in l:
            d.update(a)
        o.append(d)
    o = np.reshape(o, N).tolist()
    return o


class SubdomainScalarReal:
    def __new__(self, markers, subdomains, mapping, cpp=True, **kwargs):
        ClassReturn = _SubdomainCpp if cpp else _SubdomainPy
        return ClassReturn(markers, subdomains, mapping, **kwargs)


class SubdomainScalarComplex:
    def __new__(self, markers, subdomains, mapping, cpp=True, **kwargs):
        mapping_re, mapping_im = _separate_mapping_parts(mapping)
        re = SubdomainScalarReal(markers, subdomains, mapping_re, cpp=cpp, **kwargs)
        im = SubdomainScalarReal(markers, subdomains, mapping_im, cpp=cpp, **kwargs)
        return Complex(re, im)


class SubdomainTensorReal:
    def __new__(self, markers, subdomains, mapping, cpp=True, **kwargs):
        mapping_tensor = _make_tensor(mapping)
        mapping_list = _dic2list(mapping_tensor)
        fun = lambda mapping_list: SubdomainScalarReal(
            markers, subdomains, mapping_list, cpp=cpp, **kwargs
        )
        q = _apply(mapping_list, fun)
        return dolfin.as_tensor(q)


class SubdomainTensorComplex:
    def __new__(self, markers, subdomains, mapping, cpp=True, **kwargs):
        mapping_tensor = _make_tensor(mapping)
        d = _dic2list(mapping_tensor)
        a = _apply(d, _separate_mapping_parts)
        mape_re = _apply(a, lambda a: a[0])
        mape_im = _apply(a, lambda a: a[1])
        fun = lambda mapping_list: SubdomainScalarReal(
            markers, subdomains, mapping_list, cpp=cpp, **kwargs
        )
        qre = _apply(mape_re, fun)
        qim = _apply(mape_im, fun)
        Tre = dolfin.as_tensor(qre)
        Tim = dolfin.as_tensor(qim)
        return Complex(Tre, Tim)


class Subdomain:
    def __new__(self, markers, subdomains, mapping, cpp=True, **kwargs):
        iterable = any([isiter(v) for v in mapping.values()])
        flatvals = _flatten_list(mapping.values())
        cplx = any([iscomplex(v) and np.any(v.imag != 0) for v in flatvals])
        if iterable:
            ClassReturn = SubdomainTensorComplex if cplx else SubdomainTensorReal
        else:
            ClassReturn = SubdomainScalarComplex if cplx else SubdomainScalarReal
        return ClassReturn(markers, subdomains, mapping, cpp=cpp, **kwargs)


def _tensor_const_3d(T, real=False):
    def _treal(T):
        m = []
        for i in range(3):
            col = []
            for j in range(3):
                col.append(dolfin.Constant(T[i, j]))
            m.append(col)
        return dolfin.as_tensor(m)

    assert T.shape == (3, 3)
    if hasattr(T, "real") and hasattr(T, "imag") and not real:
        return Complex(_treal(T.real), _treal(T.imag))
    else:
        return _treal(T)


def tensor_const(T, dim=3, real=False):
    if dim == 3:
        return _tensor_const_3d(T, real=real)
    elif dim == 2:
        return _tensor_const_2d(T, real=real)
    else:
        raise NotImplementedError("only supports dim = 2 or 3")


def _check_len(p):
    if hasattr(p, "__len__"):
        try:
            lenp = len(p)
        except NotImplementedError:
            lenp = 0
        return lenp
    else:
        return 0


def _make_constant_property_3d(prop, inv=False, real=False):
    new_prop = {}
    for d, p in prop.items():
        lenp = _check_len(p)
        if lenp > 0:
            k = np.linalg.inv(np.array(p)) if inv else np.array(p)
            new_prop[d] = tensor_const(k, dim=3, real=real)
        else:
            k = 1 / p + 0j if inv else p + 0j
            if callable(k):
                new_prop[d] = k
            else:
                new_prop[d] = Constant(k)
    return new_prop


def _make_constant_property_2d(prop, inv=False, real=False):
    new_prop = {}
    for d, p in prop.items():
        lenp = _check_len(p)
        if lenp > 0:
            p = np.array(p)
            if p.shape != (2, 2):
                p = p[:2, :2]
            k = np.array(p)
            new_prop[d] = tensor_const(k, dim=2, real=real)
        else:
            k = p + 0j
            if callable(k):
                new_prop[d] = k
            else:
                new_prop[d] = Constant(k)
    return new_prop


def make_constant_property(*args, dim=3, **kwargs):
    if dim == 3:
        return _make_constant_property_3d(*args, **kwargs)
    elif dim == 2:
        return _make_constant_property_2d(*args, **kwargs)
    else:
        raise NotImplementedError("only supports dim = 2 or 3")


def _tensor_const_2d(T, real=False):
    def _treal(T):
        m = []
        for i in range(2):
            col = []
            for j in range(2):
                col.append(dolfin.Constant(T[i, j]))
            m.append(col)
        return dolfin.as_tensor(m)

    assert T.shape == (2, 2)
    if hasattr(T, "real") and hasattr(T, "imag") and not real:
        return Complex(_treal(T.real), _treal(T.imag))
    else:
        return _treal(T)


def complex_vector(V):
    return Complex(
        dolfin.as_tensor([q.real for q in V]), dolfin.as_tensor([q.imag for q in V])
    )


## xi
def _get_xi(prop):
    new_prop = {}
    for d, p in prop.items():
        lenp = _check_len(p)
        if lenp > 0:
            p = np.array(p)
            k = p[:2, :2].T
            k = k / np.linalg.det(k)
        else:
            k = 1 / p + 0j
        new_prop[d] = k
    return new_prop


## chi
def _get_chi(prop):
    new_prop = {}
    for d, p in prop.items():
        lenp = _check_len(p)
        if lenp > 0:
            p = np.array(p)
            k = p[2, 2]
        else:
            k = p + 0j
        new_prop[d] = k
    return new_prop


def _invert_3by3_complex_matrix(m):
    m1, m2, m3, m4, m5, m6, m7, m8, m9 = [m[i][j] for i in range(3) for j in range(3)]

    determinant = (
        m1 * m5 * m9
        + m4 * m8 * m3
        + m7 * m2 * m6
        - m1 * m6 * m8
        - m3 * m5 * m7
        - m2 * m4 * m9
    )
    inv = [
        [m5 * m9 - m6 * m8, m3 * m8 - m2 * m9, m2 * m6 - m3 * m5],
        [m6 * m7 - m4 * m9, m1 * m9 - m3 * m7, m3 * m4 - m1 * m6],
        [m4 * m8 - m5 * m7, m2 * m7 - m1 * m8, m1 * m5 - m2 * m4],
    ]
    # inv_df = dolfin.as_tensor(inv)
    invre = np.zeros((3, 3), dtype=object)
    invim = np.zeros((3, 3), dtype=object)
    for i in range(3):
        for j in range(3):
            q = inv[i][j]
            invre[i, j] = q.real
            invim[i, j] = q.imag
    invre = invre.tolist()
    invim = invim.tolist()

    return Complex(dolfin.as_tensor(invre), dolfin.as_tensor(invim)) / determinant


def _make_cst_mat(a, b):
    xi = _get_xi(a)
    chi = _get_chi(b)
    xi_ = make_constant_property(xi, dim=2)
    chi_ = make_constant_property(chi, dim=2)
    return xi_, chi_


def _coefs(a, b):
    # xsi = det Q^T/det Q
    extract = lambda q: dolfin.as_tensor([[q[0][0], q[1][0]], [q[0][1], q[1][1]]])
    det = lambda M: M[0][0] * M[1][1] - M[1][0] * M[0][1]
    a2 = Complex(extract(a.real), extract(a.imag))
    xi = a2 / det(a2)
    chi = b[2][2]
    return xi, chi


class Coefficient:
    def __init__(self, dict, geometry=None, pmls=[], dim=2, degree=1):
        self.dict = dict
        self.geometry = geometry
        self.pmls = pmls
        self.dim = dim
        self.degree = degree
        cell_type = "triangle" if dim == 2 else "tetrahedron"
        self.element = None  # dolfin.FiniteElement("DG", cell_type, self.degree)
        if geometry:
            if self.dim == 2:
                markers_key, mapping_key = "triangle", "surfaces"
            else:
                markers_key, mapping_key = "tetra", "volumes"

            self.markers = geometry.mesh_object["markers"][markers_key]
            self.mapping = geometry.subdomains[mapping_key]

        if pmls is not []:
            self.appy_pmls()

    def __repr__(self):
        return "Coefficient " + self.dict.__repr__()

    def build_pmls(self):
        new_material_dict = self.dict.copy()
        for pml in self.pmls:
            t = np.array(pml.transformation_matrix())
            eps_pml = (self.dict[pml.matched_domain] * t).tolist()
            new_material_dict[pml.applied_domain] = eps_pml
        return new_material_dict

    def build_annex(self, domains, reference):
        assert reference in self.dict.keys()
        annex_material_dict = self.dict.copy()
        if isinstance(domains, str):
            domains = [domains]
        for dom in domains:
            assert dom in self.dict.keys()
            annex_material_dict[dom] = self.dict[reference]
        annex_material_dict
        annex = copy.copy(self)
        annex.dict = annex_material_dict
        return annex

    def appy_pmls(self):
        self.dict.update(self.build_pmls())

    def as_subdomain(self, **kwargs):
        return Subdomain(
            self.markers,
            self.mapping,
            self.dict,
            degree=self.degree,
            element=self.element,
            **kwargs,
        )

    def as_property(self, dim=None, **kwargs):
        dim = dim or self.dim
        return make_constant_property(self.dict, dim=self.dim, **kwargs)

    def to_xi(self):
        new = copy.copy(self)
        new.dict = _get_xi(self.dict)
        return new

    def to_chi(self):
        new = copy.copy(self)
        new.dict = _get_chi(self.dict)
        return new

    def plot(self, component=None, **kwargs):

        proj_space = dolfin.FunctionSpace(self.geometry.mesh_object["mesh"], "DG", 0)
        eps_subdomain = self.as_subdomain()
        eps_plot = (
            eps_subdomain
            if component is None
            else eps_subdomain[component[0]][component[1]]
        )
        return plot(eps_plot, proj_space=proj_space, **kwargs)

    def invert(self):
        new = copy.copy(self)
        for dom, val in new.dict.items():
            val_shape = np.array(val).shape
            if val_shape == (2, 2) or val_shape == (3, 3):
                new.dict[dom] = np.linalg.inv(val)
            else:
                new.dict[dom] = 1 / val
        return new

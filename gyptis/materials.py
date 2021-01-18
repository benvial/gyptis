#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: MIT


import numpy as np

from gyptis.complex import *


class _SubdomainPy(df.UserExpression):
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


class _SubdomainCpp(df.CompiledExpression):
    def __init__(self, markers, subdomains, mapping, **kwargs):
        import os

        here = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(here, "subdomain.cpp")) as f:
            subdomain_code = f.read()
        compiled_cpp = df.compile_cpp_code(subdomain_code).SubdomainCpp()
        super().__init__(
            compiled_cpp,
            markers=markers,
            subdomains=subdomains,
            mapping=mapping,
            **kwargs
        )
        self.markers = markers
        self.subdomains = subdomains
        self.mapping = mapping


def flatten_list(k):
    result = list()
    for i in k:
        if isinstance(i, list):
            result.extend(flatten_list(i))  # Recursive call
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
        vflat = flatten_list(v)
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


class SubdomainScalarReal(object):
    def __new__(self, markers, subdomains, mapping, cpp=True, **kwargs):
        ClassReturn = _SubdomainCpp if cpp else _SubdomainPy
        return ClassReturn(markers, subdomains, mapping, **kwargs)


class SubdomainScalarComplex(object):
    def __new__(self, markers, subdomains, mapping, cpp=True, **kwargs):
        mapping_re, mapping_im = _separate_mapping_parts(mapping)
        re = SubdomainScalarReal(markers, subdomains, mapping_re, cpp=cpp, **kwargs)
        im = SubdomainScalarReal(markers, subdomains, mapping_im, cpp=cpp, **kwargs)
        return Complex(re, im)


class SubdomainTensorReal(object):
    def __new__(self, markers, subdomains, mapping, cpp=True, **kwargs):
        mapping_tensor = _make_tensor(mapping)
        mapping_list = _dic2list(mapping_tensor)
        fun = lambda mapping_list: SubdomainScalarReal(
            markers, subdomains, mapping_list, cpp=cpp, **kwargs
        )
        q = _apply(mapping_list, fun)
        return df.as_tensor(q)


class SubdomainTensorComplex(object):
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
        Tre = df.as_tensor(qre)
        Tim = df.as_tensor(qim)
        return Complex(Tre, Tim)


class Subdomain(object):
    def __new__(self, markers, subdomains, mapping, cpp=True, **kwargs):
        iterable = any([isiter(v) for v in mapping.values()])
        flatvals = flatten_list(mapping.values())
        cplx = any([iscomplex(v) and np.any(v.imag != 0) for v in flatvals])
        if iterable:
            ClassReturn = SubdomainTensorComplex if cplx else SubdomainTensorReal
        else:
            ClassReturn = SubdomainScalarComplex if cplx else SubdomainScalarReal
        return ClassReturn(markers, subdomains, mapping, cpp=cpp, **kwargs)


def tensor_const(T):
    def _treal(T):
        m = []
        for i in range(3):
            col = []
            for j in range(3):
                col.append(df.Constant(T[i, j]))
            m.append(col)
        return df.as_tensor(m)

    assert T.shape == (3, 3)
    if hasattr(T, "real") and hasattr(T, "imag"):
        return Complex(_treal(T.real), _treal(T.imag))
    else:
        return _treal(T)


def make_constant_property(prop, inv=False):
    new_prop = {}
    for d, p in prop.items():
        if hasattr(p, "__len__") and len(p) > 1:
            k = np.linalg.inv(np.array(p)) if inv else np.array(p)
            new_prop[d] = tensor_const(k)
        else:
            k = 1 / p + 0j if inv else p + 0j
            new_prop[d] = Constant(k)
    return new_prop


def tensor_const_2d(T):
    def _treal(T):
        m = []
        for i in range(2):
            col = []
            for j in range(2):
                col.append(df.Constant(T[i, j]))
            m.append(col)
        return df.as_tensor(m)

    assert T.shape == (2, 2)
    if hasattr(T, "real") and hasattr(T, "imag"):
        return Complex(_treal(T.real), _treal(T.imag))
    else:
        return _treal(T)


def make_constant_property_2d(prop):
    new_prop = {}
    for d, p in prop.items():
        if hasattr(p, "__len__"):
            try:
                lenp = len(p)
            except NotImplementedError:
                lenp = 1
        if hasattr(p, "__len__") and lenp > 1:
            p = np.array(p)
            if p.shape != (2, 2):
                p = p[:2, :2]
            k = np.array(p)
            new_prop[d] = tensor_const_2d(k)
        else:
            k = p + 0j
            if callable(k):
                new_prop[d] = k
            else:
                new_prop[d] = Constant(k)
    return new_prop


def complex_vector(V):
    return Complex(df.as_tensor([q.real for q in V]), df.as_tensor([q.imag for q in V]))


## xi
def get_xi(prop):
    new_prop = {}
    for d, p in prop.items():
        if hasattr(p, "__len__"):
            try:
                lenp = len(p)
            except NotImplementedError:
                lenp = 1
        if hasattr(p, "__len__") and lenp > 1:
            p = np.array(p)
            k = p[:2, :2].T
            k /= np.linalg.det(k)
        else:
            k = 1 / p + 0j
        new_prop[d] = k
    return new_prop


## chi
def get_chi(prop):
    new_prop = {}
    for d, p in prop.items():
        if hasattr(p, "__len__"):
            try:
                lenp = len(p)
            except NotImplementedError:
                lenp = 1
        if hasattr(p, "__len__") and lenp > 1:
            p = np.array(p)
            k = p[2, 2]
        else:
            k = p + 0j
        new_prop[d] = k
    return new_prop


#
#
# xi = get_xi(g.epsilon)
# chi = get_chi(g.mu)
#
#
# xi_ = make_constant_property_2d(xi)
# chi_ = make_constant_property_2d(chi)

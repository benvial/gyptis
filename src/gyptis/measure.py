#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# License: MIT
# See the documentation at gyptis.gitlab.io


from . import dolfin


class Measure(dolfin.Measure):
    def __init__(
        self,
        integral_type,
        domain=None,
        subdomain_id="everywhere",
        metadata=None,
        subdomain_data=None,
        subdomain_dict=None,
    ):

        self.subdomain_dict = subdomain_dict
        if (
            self.subdomain_dict
            and isinstance(subdomain_id, str)
            and subdomain_id != "everywhere"
        ):
            subdomain_id = self.subdomain_dict[subdomain_id]
        super().__init__(
            integral_type,
            domain=domain,
            subdomain_id=subdomain_id,
            metadata=metadata,
            subdomain_data=subdomain_data,
        )

    def __call_single__(self, subdomain_id=None, **kwargs):
        if (
            self.subdomain_dict
            and isinstance(subdomain_id, str)
            and subdomain_id != "everywhere"
        ):
            subdomain_id = self.subdomain_dict[subdomain_id]
        return super().__call__(subdomain_id=subdomain_id, **kwargs)

    def __call__(self, subdomain_id=None, **kwargs):
        subdomain_id = None if subdomain_id == [] else subdomain_id
        if isinstance(subdomain_id, list):
            for i, sid in enumerate(subdomain_id):
                if i == 0:
                    out = self.__call_single__(subdomain_id=sid, **kwargs)
                else:
                    out += self.__call_single__(subdomain_id=sid, **kwargs)
            return out
        else:
            return self.__call_single__(subdomain_id=subdomain_id, **kwargs)

from .__about__ import (
    __author__,
    __author_email__,
    __copyright__,
    __description__,
    __license__,
    __status__,
    __version__,
    __website__,
)

__doc__ = __description__

import os

import dolfin

if os.environ.get("GYPTIS_ADJOINT") is not None:
    import dolfin_adjoint

    ADJOINT = True
    dolfin.__dict__.update(dolfin_adjoint.__dict__)
    dolfin.__spec__.name = "dolfin"
    del dolfin_adjoint
else:
    ADJOINT = False

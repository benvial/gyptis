
Gyptis
======

:Release: |release|
:Date: |today|

Gyptis is an open-source Python package to solve problems 
in Photonics and Electromagnetism using the Finite Element method. 
It relies on Gmsh_ for geometry definition and meshing 
and Fenics_ for solving Maxwell's equations.


.. _Gmsh: https://gmsh.info/
.. _Fenics: https://fenicsproject.org/


.. .. toctree::
..    :maxdepth: 1
..
..    install_upgrade
..    api
..    release

Examples
--------

Tutorials with worked examples and background information for
submodules.

.. toctree::
   :maxdepth: 2

   auto_examples/index.rst




API Reference
-------------

:mod:`gyptis.geometry`: Defining geometry and generating meshes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: gyptis.geometry
    :no-members:
    :no-inherited-members:

Base classes
""""""""""""
.. currentmodule:: gyptis

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst
   
   geometry.Geometry
   geometry.BoxPML
   geometry.BoxPML3D



-------------------------------------



:mod:`gyptis.complex`: Complex numbers support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: gyptis.complex
    :no-members:
    :no-inherited-members:

Base classes
""""""""""""
.. currentmodule:: gyptis

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class_with_call.rst
   
   complex.Complex


Functions
"""""""""
.. currentmodule:: gyptis

.. autosummary::
  :nosignatures:
  :toctree: generated/
  :template: function.rst
  
  complex.iscomplex


-------------------------------------

:mod:`gyptis.grating_2d`: Two dimensional diffraction gratings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: gyptis.grating_2d
    :no-members:
    :no-inherited-members:

Base classes
""""""""""""
.. currentmodule:: gyptis

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst
   
   grating_2d.Layered2D
   grating_2d.Grating2D

-------------------------------------


Indices and search
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Gyptis
======

:Release: |release|
:Date: |today|

Gyptis is an open-source Python package to solve problems 
in Photonics and Electromagnetism using the Finite Element method. 
It relies on ``gmsh`` for geometry definition and meshing 
and ``fenics`` for solving Maxwell's equations.

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

The exact API of all functions and classes, as given by the docstrings. The API
documents expected types and allowed features for all functions, and all
parameters available for the algorithms.


.. 
.. Functions
.. ---------
.. .. currentmodule:: sklearn
.. 
.. .. autosummary::
..    :toctree: generated/
..    :template: function.rst
.. 
..    base.clone
..    base.is_classifier
..    base.is_regressor
..    config_context
..    get_config
..    set_config
..    show_versions



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


Indices and search
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# This file is part of gyptis
# Version: 1.0.2
# License: MIT
# See the documentation at gyptis.gitlab.io


import json
import os
from urllib.request import urlretrieve


def _get_data_path(data_path=None):
    """Return path to data dir.

    This directory stores large datasets required for the examples, to avoid
    downloading the data several times.

    By default the data dir is set to a folder named '.gyptis/data' in the
    user home folder.

    If the folder does not already exist, it is automatically created.

    Parameters
    ----------
    data_path : str | None
        The full path to the data dir. ``~/.gyptis/data`` by default.
    """
    if data_path is None:
        data_path = os.path.join("~", ".gyptis/data")
        data_path = os.path.expanduser(data_path)

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    return data_path


def _get_config_path(config_path):
    """Return path to config file

    Parameters
    ----------
    config_path : str | None
        The path to the data dir. ``~/.gyptis`` by default.
    """
    if config_path is None:
        config_path = os.path.join("~", ".gyptis")
        config_path = os.path.expanduser(config_path)
    else:
        config_path = os.path.join(config_path, ".gyptis")
    return config_path


def _load_config(config_path):
    """Safely load a config file."""
    with open(config_path, "r") as fid:
        try:
            config = json.load(fid)
        except ValueError as e:
            # No JSON object could be decoded --> corrupt file?
            msg = f"The config file ({config_path}) is not a valid JSON file and might be corrupted"
            raise RuntimeError(msg) from e
    return config


def _set_config(config, key, value, config_file):
    """Set the configurations in the config file.

    Parameters
    ----------
    key : str
        The preference key to set.
    value : str |  None
        The value to assign to the preference key. If None, the key is
        deleted.
    config_path : str | None
        The path to the .gyptis directory.
    """
    if not isinstance(key, str):
        raise TypeError(f"key must be of type str, got {type(key)} instead")
    if not isinstance(value, str):
        raise TypeError(f"value must be of type str, got {type(value)} instead")

    if value is None:
        config.pop(key, None)
    else:
        config[key] = value

    # Write all values. This may fail if the default directory is not
    # writeable.
    config_path = os.path.dirname(config_file)
    if not os.path.isdir(config_path):
        os.mkdir(config_path)
    with open(config_file, "w") as fid:
        json.dump(config, fid, sort_keys=True, indent=0)


def download_data(
    url, data_file_name, data_key="data_dir", data_path=None, config_path=None
):
    """Downloads a remote dataset and saves path to config file.

    Checks if the data file already exists in either the path saved under the
    key ``data_key`` in the config file or the default data path;
    ``~/.gyptis/data``. If the file does not exist, downloads the data
    from ``url`` and saves to ``data_file_name`` in data_path. Finally, stores
    the location of the data in a config file, under key ``data_key``. Returns
    the path to the data file.

    Parameters
    ----------
    url : str
        Dataset URL.

    data_file_name : str
        File name to save the dataset at.


    data_path : str | None
        The path to the data dir. ``~/.gyptis/data`` by default.

    config_path: str | None
        The path to the config file. ``~/.gyptis`` by default.

    Returns
    -------
    data_file : str
        Full path of the created file.
    """

    config_path = _get_config_path(config_path)
    config_file = os.path.join(config_path, "gyptis_config.json")
    config = _load_config(config_file) if os.path.isfile(config_file) else {}
    data_path = config.get(data_key, None) or _get_data_path(data_path=data_path)
    data_file = os.path.join(data_path, data_file_name)
    # Download file if it doesn't exist
    if not os.path.exists(data_file):
        urlretrieve(url, data_file)
    # save download location in config
    _set_config(config, data_key, data_path, config_file)
    return data_file


def download_example_data(
    data_file_name, example_dir, data_key="data_dir", data_path=None, config_path=None
):
    url = (
        "https://gitlab.com/gyptis/gyptis/-/raw/master/examples/"
        + example_dir
        + "/"
        + data_file_name
    )

    return download_data(
        url,
        data_file_name,
        data_key=data_key,
        data_path=data_path,
        config_path=config_path,
    )

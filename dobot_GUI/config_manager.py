"""Read and write settings stored in ``config.ini``.

All other modules get their tunable values (robot IP, camera options, pick
height, ...) through the helpers in this file instead of hard-coding them.
Each getter re-reads the file on every call, so an edit to ``config.ini`` takes
effect the next time a value is requested, without restarting the GUI.

Example:
    >>> from config_manager import get_float
    >>> get_float("Pick", "PICK_Z", 10.0)
    10.0
"""

import configparser
import os
from configupdater import ConfigUpdater

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini")

config = configparser.ConfigParser(allow_no_value=True)


def get_str(section, key, fallback=None):
    """Return a string value from the config file.

    Args:
        section: Name of the ``[section]`` in ``config.ini``.
        key: Name of the key inside the section.
        fallback: Value returned if the section or key is missing.

    Returns:
        The stored string, or ``fallback`` if it is not found.
    """
    config.read(CONFIG_PATH, encoding="utf-8")
    return config.get(section, key, fallback=fallback)


def get_float(section, key, fallback=None):
    """Return a value from the config file parsed as a float.

    Args:
        section: Name of the ``[section]`` in ``config.ini``.
        key: Name of the key inside the section.
        fallback: Value returned if the section or key is missing.

    Returns:
        The stored value as a float, or ``fallback`` if it is not found.

    Raises:
        ValueError: If the stored value is present but is not a number.
    """
    config.read(CONFIG_PATH, encoding="utf-8")
    return config.getfloat(section, key, fallback=fallback)


def get_int(section, key, fallback=None):
    """Return a value from the config file parsed as an integer.

    Args:
        section: Name of the ``[section]`` in ``config.ini``.
        key: Name of the key inside the section.
        fallback: Value returned if the section or key is missing.

    Returns:
        The stored value as an int, or ``fallback`` if it is not found.

    Raises:
        ValueError: If the stored value is present but is not an integer.
    """
    config.read(CONFIG_PATH, encoding="utf-8")
    return config.getint(section, key, fallback=fallback)


def get_bool(section, key, fallback=None):
    """Return a value from the config file parsed as a boolean.

    Accepts the values ``1``/``0``, ``yes``/``no``, ``true``/``false`` and
    ``on``/``off`` (case-insensitive).

    Args:
        section: Name of the ``[section]`` in ``config.ini``.
        key: Name of the key inside the section.
        fallback: Value returned if the section or key is missing.

    Returns:
        The stored value as a bool, or ``fallback`` if it is not found.

    Raises:
        ValueError: If the stored value is not a recognised boolean.
    """
    config.read(CONFIG_PATH, encoding="utf-8")
    return config.getboolean(section, key, fallback=fallback)


def set_value(section, key, value):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.lstrip().startswith(f"{key} ") or line.lstrip().startswith(f"{key}="):
            prefix = line.split("=")[0]
            lines[i] = f"{prefix}= {value}\n"
            break

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)
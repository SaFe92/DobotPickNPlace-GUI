import configparser
import os
from configupdater import ConfigUpdater

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini")

config = configparser.ConfigParser(allow_no_value=True)

def get_str(section, key, fallback=None):
    config.read(CONFIG_PATH, encoding="utf-8")
    return config.get(section, key, fallback=fallback)

def get_float(section, key, fallback=None):
    config.read(CONFIG_PATH, encoding="utf-8")
    return config.getfloat(section, key, fallback=fallback)

def get_int(section, key, fallback=None):
    config.read(CONFIG_PATH, encoding="utf-8")
    return config.getint(section, key, fallback=fallback)

def get_bool(section, key, fallback=None):
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
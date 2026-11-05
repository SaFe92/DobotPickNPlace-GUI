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
    updater = ConfigUpdater()
    updater.read(CONFIG_PATH, encoding="utf-8")

    if section not in updater:
        updater.add_section(section)

    updater[section][key] = str(value)

    with open(CONFIG_PATH, "w", encoding="utf-8") as configfile:
        updater.write(configfile)
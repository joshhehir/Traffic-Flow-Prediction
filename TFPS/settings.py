import json

"""Adds the settings into a directory"""
settings: {}
with open('config.json', 'r') as f:
    settings = json.load(f)


def get_setting(key):
    """Gets the requested settings key and returns a string"""
    return settings[key]

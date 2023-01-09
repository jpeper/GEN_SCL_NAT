"""
Module containing helper functions for GEN_SCL_NAT project
"""

def load_mappings():
    """
    Load category mappings used to map existing labelset to human-readable variant
    """
    import os
    import json
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, 'category_mappings.json')) as ofile:
        data_json = json.load(ofile)
    return data_json

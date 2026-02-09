"""
Separate module to test Mapping type resolution.
This simulates how thinc's premap_ids.pyx is in a separate module.
"""
from typing import Mapping

from confection.tests.util import my_registry


@my_registry.cats("separate_mapping_cat.v1")
def separate_mapping_cat(mapping_table: Mapping[int, int], default: int = 0) -> str:
    """Function with Mapping parameter defined in a separate module."""
    return f"mapping with {len(mapping_table)} items, default={default}"

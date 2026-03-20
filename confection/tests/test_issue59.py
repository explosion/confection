"""Test for issue #59: Registered functions with pydantic class arguments
should receive constructed model instances, not have them flattened to dicts.

This is closely related to #58 but tests the case where the BaseModel arg
is itself resolved from a registered function (nested promise).
"""

import sys

import catalogue
import pytest

pydantic = pytest.importorskip("pydantic")
if sys.version_info >= (3, 14):
    pytest.skip(
        "pydantic v1 is not compatible with Python 3.14+", allow_module_level=True
    )

try:
    from pydantic.v1 import BaseModel
except ImportError:
    from pydantic import BaseModel  # type: ignore

from confection import Config, registry


class Dim(BaseModel):
    length: int


@pytest.fixture(autouse=True)
def _register():
    if not hasattr(registry, "dims"):
        registry.dims = catalogue.create(
            "confection", "dims", entry_points=False
        )
    if not hasattr(registry, "architectures"):
        registry.architectures = catalogue.create(
            "confection", "architectures", entry_points=False
        )
    registry.dims.register("Dim.v1", func=make_dim)
    registry.architectures.register("arch.v1", func=load_architecture)
    yield


def make_dim(length: int) -> Dim:
    return Dim(length=length)


def load_architecture(foo: str, dim: Dim) -> str:
    # dim should be a Dim instance, not a dict
    return f"{foo}:{dim.length}"


def test_nested_basemodel_arg():
    """A registered function receiving a BaseModel resolved from another
    registered function should get the model instance."""
    config = Config().from_str(
        """
[model]
@architectures = "arch.v1"
foo = "test"

[model.dim]
@dims = "Dim.v1"
length = 6
"""
    )
    result = registry.resolve(config)
    assert result["model"] == "test:6"


def test_direct_basemodel_arg():
    """A registered function receiving a BaseModel from config keys
    (not a promise) should get the model instance."""
    config = Config().from_str(
        """
[model]
@architectures = "arch.v1"
foo = "test"

[model.dim]
length = 6
"""
    )
    result = registry.resolve(config)
    assert result["model"] == "test:6"

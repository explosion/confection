"""Test for issue #58: Registered functions with pydantic BaseModel arguments
should receive constructed model instances, not raw dicts.
"""

import sys

import catalogue
import pytest

pydantic = pytest.importorskip("pydantic")
if sys.version_info >= (3, 14):
    pytest.skip("pydantic v1 is not compatible with Python 3.14+", allow_module_level=True)

try:
    from pydantic.v1 import BaseModel
except ImportError:
    from pydantic import BaseModel  # type: ignore

from confection import Config, registry


class Resource(BaseModel):
    url: str


@pytest.fixture(autouse=True)
def _register():
    if not hasattr(registry, "resources"):
        registry.resources = catalogue.create(
            "confection", "resources", entry_points=False
        )
    registry.resources.register("Resource.v1", func=make_resource)
    yield


def make_resource(resource: Resource) -> str:
    return resource.url


def test_basemodel_arg_receives_model_instance():
    """A registered function with a BaseModel param should receive
    the constructed model, not a plain dict."""
    config = Config().from_str(
        """
[test]
@resources = "Resource.v1"

[test.resource]
url = "http://example.com"
"""
    )
    result = registry.resolve(config)
    assert result["test"] == "http://example.com"

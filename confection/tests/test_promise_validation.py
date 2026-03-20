"""Test that promise dicts validate correctly against Union[Callable, Promise]."""

import pytest

from confection._registry import Promise
from confection.validation import validate_type

pydantic_v1 = pytest.importorskip("pydantic.v1")
BaseModel = pydantic_v1.BaseModel
create_model = pydantic_v1.create_model

from typing import Callable, Dict, Iterable, Union


# The annotation spaCy uses
Reader = Union[Callable, Promise]


@pytest.fixture
def promise_dict():
    return {"@readers": "spacy.Corpus.v1", "path": "/tmp/x", "gold_preproc": False}


@pytest.fixture
def corpora_value(promise_dict):
    return {"dev": promise_dict, "train": promise_dict}


@pytest.fixture
def annotation():
    return Dict[str, Reader]


def test_pydantic_v1_rejects_promise_dict(corpora_value, annotation):
    """Pydantic v1 also rejects raw promise dicts for Dict[str, Union[Callable, Promise]].

    Neither pydantic nor our validator should accept this — the unresolved
    schema builder replaces these annotations before validation runs.
    """
    Model = create_model(
        "Test",
        corpora=(annotation, ...),
        __config__=type("Config", (), {"arbitrary_types_allowed": True}),
    )
    with pytest.raises(pydantic_v1.ValidationError):
        Model(corpora=corpora_value)


def test_validate_type_rejects_promise_dict(corpora_value, annotation):
    """Our validate_type also rejects this (consistent with pydantic v1)."""
    err = validate_type(corpora_value, annotation)
    assert err is not None

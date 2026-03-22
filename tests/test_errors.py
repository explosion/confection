"""Test _errors.py edge cases."""
from confection._errors import ConfigValidationError


def test_from_error():
    original = ConfigValidationError(
        errors=[{"loc": ["a"], "msg": "bad"}],
        title="Original",
        desc="original desc",
    )
    new = ConfigValidationError.from_error(original, title="New title")
    assert new.title == "New title"
    assert new.desc == "original desc"
    assert new.errors == original.errors


def test_format_with_parent():
    err = ConfigValidationError(
        errors=[{"loc": ["x"], "msg": "bad"}],
        title="Error",
        parent="section",
    )
    assert "section" in err.text
    assert "-> x" in err.text

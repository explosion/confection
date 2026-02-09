"""Systematic tests for config value types using Hypothesis.

Uses property-based testing to explore the space of possible config values.
"""

import pytest
import srsly
from hypothesis import given, strategies as st, settings, example, HealthCheck
from numpy.testing import assert_equal, assert_allclose
from confection import Config
from confection._config import try_load_json
from confection._registry import make_func_schema, get_func_fields
from confection.tests.util import my_registry
from pydantic import ValidationError


# =============================================================================
# Unit tests for try_load_json - the core parsing function
# =============================================================================
# Contract: parse as JSON if valid, otherwise return original string


class TestTryLoadJson:
    """Test the core JSON parsing function."""

    # Valid JSON literals -> parsed Python values
    @pytest.mark.parametrize(
        "inp,expected",
        [
            ("42", 42),
            ("-42", -42),
            ("3.14", 3.14),
            ("-3.14", -3.14),
            ("0", 0),
            ("true", True),
            ("false", False),
            ("null", None),
            ("[1, 2, 3]", [1, 2, 3]),
            ('{"a": 1}', {"a": 1}),
            ("[]", []),
            ("{}", {}),
        ],
    )
    def test_json_literals(self, inp, expected):
        """Valid JSON literals are parsed to Python values."""
        assert try_load_json(inp) == expected

    # Quoted strings -> unquoted Python strings
    @pytest.mark.parametrize(
        "inp,expected",
        [
            ('"hello"', "hello"),
            ('"with spaces"', "with spaces"),
            ('""', ""),
            ('"0"', "0"),  # Quoted "0" should be string, not int
            ('"-42"', "-42"),  # Quoted "-42" should be string, not int
            ('"true"', "true"),  # Quoted "true" should be string, not bool
            ('"false"', "false"),
            ('"null"', "null"),
            ('"3.14"', "3.14"),  # Quoted "3.14" should be string, not float
        ],
    )
    def test_quoted_strings(self, inp, expected):
        """Quoted strings are unquoted to Python strings."""
        assert try_load_json(inp) == expected

    # Invalid JSON -> returned as-is
    @pytest.mark.parametrize(
        "inp",
        [
            "hello",  # unquoted string
            "hello world",  # unquoted with space
            "not json",
            "${var.ref}",  # variable reference
            "hello ${var}",  # string with variable
        ],
    )
    def test_invalid_json_returned_as_is(self, inp):
        """Invalid JSON strings are returned unchanged."""
        assert try_load_json(inp) == inp


# =============================================================================
# Unit tests for CustomInterpolation.before_read
# =============================================================================
# This is where the bug lives. before_read receives raw INI values and
# preprocesses them before interpolation.

# =============================================================================
# Tests for parsing with plain ExtendedInterpolation (no CustomInterpolation)
# =============================================================================
# These tests define what SHOULD work if we remove the buggy CustomInterpolation

from configparser import ConfigParser, ExtendedInterpolation


class TestPlainExtendedInterpolation:
    """Test parsing with plain ExtendedInterpolation (no custom before_read)."""

    def _parse(self, config_str):
        """Parse a config string using plain ExtendedInterpolation."""
        parser = ConfigParser(interpolation=ExtendedInterpolation())
        parser.read_string(config_str)
        return parser

    def _parse_value(self, raw_value):
        """Parse a single raw value through try_load_json."""
        return try_load_json(raw_value)

    # Basic value types
    @pytest.mark.parametrize(
        "ini_value,expected_type,expected_value",
        [
            ("42", int, 42),
            ("-42", int, -42),
            ("3.14", float, 3.14),
            ("true", bool, True),
            ("false", bool, False),
            ("null", type(None), None),
            ("[1, 2, 3]", list, [1, 2, 3]),
            ('{"a": 1}', dict, {"a": 1}),
        ],
    )
    def test_unquoted_json_literals(self, ini_value, expected_type, expected_value):
        """Unquoted JSON literals parse to their Python types."""
        parser = self._parse(f"[s]\nv = {ini_value}")
        raw = parser.get("s", "v")
        parsed = self._parse_value(raw)
        assert type(parsed) == expected_type
        assert parsed == expected_value

    # Quoted strings - the key test cases
    @pytest.mark.parametrize(
        "ini_value,expected",
        [
            ('"hello"', "hello"),
            ('"with spaces"', "with spaces"),
            ('""', ""),
            ('"0"', "0"),  # Must stay string, not become int
            ('"-42"', "-42"),  # Must stay string, not become int
            ('"3.14"', "3.14"),  # Must stay string, not become float
            ('"true"', "true"),  # Must stay string, not become bool
            ('"false"', "false"),
            ('"null"', "null"),  # Must stay string, not become None
        ],
    )
    def test_quoted_strings_stay_strings(self, ini_value, expected):
        """Quoted strings must parse to Python strings, not other types."""
        parser = self._parse(f"[s]\nv = {ini_value}")
        raw = parser.get("s", "v")
        parsed = self._parse_value(raw)
        assert isinstance(parsed, str), f"Expected str, got {type(parsed).__name__}"
        assert parsed == expected

    # Value interpolation
    def test_value_interpolation(self):
        """Value interpolation ${section:key} should work."""
        config_str = """
[vars]
x = 10
name = "hello"

[section]
a = ${vars:x}
b = ${vars:name}
"""
        parser = self._parse(config_str)
        assert self._parse_value(parser.get("section", "a")) == 10
        assert self._parse_value(parser.get("section", "b")) == "hello"


def assert_values_equal(actual, expected):
    """Assert values are equal, using approximate comparison for floats.

    Note: Very small floats may round to 0 due to srsly serialization limits.
    """
    if isinstance(expected, float):
        # Use both relative and absolute tolerance
        # atol handles small numbers that round to 0
        assert_allclose(actual, expected, rtol=1e-5, atol=1e-10)
    elif isinstance(expected, dict):
        assert set(actual.keys()) == set(expected.keys())
        for k in expected:
            assert_values_equal(actual[k], expected[k])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert_values_equal(a, e)
    else:
        assert_equal(actual, expected)


# =============================================================================
# Strategies for config values
# =============================================================================

# Field names: valid Python identifiers, not starting with @
field_names = st.from_regex(r"[a-z][a-z0-9_]{0,10}", fullmatch=True)

# Scalar values
scalar_values = st.one_of(
    st.text(
        min_size=0, max_size=20, alphabet=st.characters(blacklist_categories=["Cs"])
    ),
    st.integers(min_value=-1000000, max_value=1000000),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    st.booleans(),
)

# Alphabet for dictionary keys - alphanumeric plus underscore, safe for JSON
DICT_KEY_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"


# Recursive strategy for nested values (lists, dicts)
def config_values():
    """Strategy for any value that can appear in a config."""
    return st.recursive(
        scalar_values,
        lambda children: st.one_of(
            st.lists(children, max_size=3),
            st.dictionaries(
                st.text(min_size=1, max_size=8, alphabet=DICT_KEY_ALPHABET),
                children,
                max_size=3,
            ),
        ),
        max_leaves=10,
    )


# A config section: dict with string field names and config values
config_section = st.dictionaries(
    field_names,
    config_values(),
    min_size=1,
    max_size=5,
)

# Section names for nested sections (no dots allowed in individual names)
section_names = st.from_regex(r"[a-z][a-z0-9_]{0,8}", fullmatch=True)


@st.composite
def nested_config(draw):
    """Generate a config with nested sections like [outer.inner.deep]."""
    # Generate 1-3 levels of nesting
    depth = draw(st.integers(min_value=1, max_value=3))
    path = [draw(section_names) for _ in range(depth)]

    # Generate content for the deepest section
    content = draw(config_section)

    # Build nested structure from inside out
    result = content
    for name in reversed(path):
        result = {name: result}

    return result, path, content


@st.composite
def config_with_positional_args(draw):
    """Generate a config with positional args using [section.*.name] syntax.

    Creates a section with 1-3 positional arg subsections that become a tuple.
    Example:
        [parent]
        key = 1

        [parent.*.first]
        x = 10

        [parent.*.second]
        y = 20

    Results in: {"parent": {"key": 1, "*": ({"x": 10}, {"y": 20})}}
    """
    parent_name = draw(section_names)

    # Parent section needs at least one field for the [parent] section to be created
    parent_fields = draw(
        st.dictionaries(
            field_names,
            st.one_of(
                st.integers(min_value=-100, max_value=100),
                st.text(min_size=1, max_size=10, alphabet=DICT_KEY_ALPHABET),
            ),
            min_size=1,
            max_size=3,
        )
    )

    # Generate 1-3 positional arg sections with unique names
    # Use a fixed pool of names to avoid expensive uniqueness checks
    positional_name_pool = ["pos1", "pos2", "pos3", "item1", "item2", "item3"]
    num_positional = draw(st.integers(min_value=1, max_value=3))
    positional_names = draw(
        st.permutations(positional_name_pool).map(lambda x: list(x)[:num_positional])
    )

    positional_contents = []
    for _ in positional_names:
        content = draw(
            st.dictionaries(
                field_names,
                st.one_of(
                    st.integers(min_value=-100, max_value=100),
                    st.booleans(),
                ),
                min_size=1,
                max_size=3,
            )
        )
        positional_contents.append(content)

    # Build expected result - the "*" key stores a dict with names as keys
    # (it becomes a tuple only during registry resolve())
    expected = dict(parent_fields)
    expected["*"] = {
        name: content for name, content in zip(positional_names, positional_contents)
    }

    return (
        parent_name,
        positional_names,
        parent_fields,
        positional_contents,
        {parent_name: expected},
    )


@st.composite
def config_with_interpolation(draw):
    """Generate a config with variable interpolation.

    Creates a source section with values and a target section that
    references some of those values via ${source.key} syntax.
    """
    # Generate source section with scalar values only (for simplicity)
    source_fields = draw(
        st.dictionaries(
            field_names,
            st.one_of(
                st.integers(min_value=-1000, max_value=1000),
                st.text(min_size=1, max_size=10, alphabet=DICT_KEY_ALPHABET),
            ),
            min_size=1,
            max_size=5,
        )
    )

    # Pick which fields to reference
    source_keys = list(source_fields.keys())
    num_refs = draw(st.integers(min_value=1, max_value=len(source_keys)))
    ref_keys = draw(st.permutations(source_keys))[:num_refs]

    # Build target section with references, tracking expected values
    target_fields = {}
    expected_target = {}
    for source_key in ref_keys:
        target_field = draw(field_names)
        target_fields[target_field] = f"${{source.{source_key}}}"
        expected_target[target_field] = source_fields[source_key]

    # Build the config dict (uninterpolated)
    config = {
        "source": source_fields,
        "target": target_fields,
    }

    return config, expected_target


# =============================================================================
# Config String Strategy - generates INI-format config strings directly
# =============================================================================

# Values that can appear in a config string (INI format)
# These are the literal string representations, not Python values
ini_string_values = st.text(
    min_size=1,
    max_size=20,
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_- ",
)
ini_int_values = st.integers(min_value=-10000, max_value=10000).map(str)
ini_float_values = st.floats(
    allow_nan=False, allow_infinity=False, min_value=-1000, max_value=1000
).map(lambda x: f"{x:.6g}")
ini_bool_values = st.sampled_from(["true", "false"])

# A single value in INI format
ini_scalar_value = st.one_of(
    # Quoted strings
    ini_string_values.map(lambda s: srsly.json_dumps(s)),
    # Unquoted numbers
    ini_int_values,
    ini_float_values,
    # Booleans
    ini_bool_values,
)

# A list in INI format: [val1, val2, ...]
ini_list_value = st.lists(ini_scalar_value, min_size=0, max_size=5).map(
    lambda items: "[" + ", ".join(items) + "]"
)

# Any value in INI format
ini_value = st.one_of(ini_scalar_value, ini_list_value)


@st.composite
def config_string(draw):
    """Generate a config string in INI format.

    Returns (config_str, expected_dict) where expected_dict is the parsed form.
    """
    # Generate 1-3 sections
    num_sections = draw(st.integers(min_value=1, max_value=3))
    sections = []
    expected = {}

    for _ in range(num_sections):
        section_name = draw(section_names)
        # Ensure unique section names
        while section_name in expected:
            section_name = draw(section_names)

        # Generate 1-5 fields per section
        num_fields = draw(st.integers(min_value=1, max_value=5))
        fields = []
        section_expected = {}

        for _ in range(num_fields):
            field_name = draw(field_names)
            # Ensure unique field names within section
            while field_name in section_expected:
                field_name = draw(field_names)

            # Choose value type and generate both string and expected value
            value_type = draw(
                st.sampled_from(["string", "int", "float", "bool", "list"])
            )

            if value_type == "string":
                py_value = draw(ini_string_values)
                ini_str = srsly.json_dumps(py_value)
            elif value_type == "int":
                py_value = draw(st.integers(min_value=-10000, max_value=10000))
                ini_str = str(py_value)
            elif value_type == "float":
                py_value = draw(
                    st.floats(
                        allow_nan=False,
                        allow_infinity=False,
                        min_value=-1000,
                        max_value=1000,
                    )
                )
                ini_str = f"{py_value:.6g}"
            elif value_type == "bool":
                py_value = draw(st.booleans())
                ini_str = "true" if py_value else "false"
            else:  # list
                list_len = draw(st.integers(min_value=0, max_value=3))
                py_value = [
                    draw(st.integers(min_value=-100, max_value=100))
                    for _ in range(list_len)
                ]
                ini_str = "[" + ", ".join(str(x) for x in py_value) + "]"

            fields.append(f"{field_name} = {ini_str}")
            section_expected[field_name] = py_value

        section_str = f"[{section_name}]\n" + "\n".join(fields)
        sections.append(section_str)
        expected[section_name] = section_expected

    config_str = "\n\n".join(sections)
    return config_str, expected


# =============================================================================
# Tests
# =============================================================================


@given(section=config_section)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_config_section_roundtrip(section):
    """Test that a config section survives being converted to string and back."""
    cfg = Config({"section": section})
    config_str = cfg.to_str()
    parsed = Config().from_str(config_str)
    assert_values_equal(parsed["section"], section)


# Scalar values that are safe for roundtrip (exclude strings with problematic patterns)
safe_scalar_values = st.one_of(
    # Exclude strings that look like JSON primitives with whitespace
    st.text(min_size=0, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz_"),
    st.integers(min_value=-1000000, max_value=1000000),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    st.booleans(),
)


@given(value=safe_scalar_values)
@settings(max_examples=100)
def test_scalar_value_roundtrip(value):
    """Test that scalar values roundtrip correctly."""
    cfg = Config({"section": {"field": value}})
    config_str = cfg.to_str()
    parsed = Config().from_str(config_str)
    assert_values_equal(parsed["section"]["field"], value)


@given(items=st.lists(scalar_values, max_size=10))
@settings(max_examples=100)
def test_list_value_roundtrip(items):
    """Test that list values roundtrip correctly."""
    cfg = Config({"section": {"field": items}})
    config_str = cfg.to_str()
    parsed = Config().from_str(config_str)
    assert_values_equal(parsed["section"]["field"], items)


@given(
    mapping=st.dictionaries(
        st.text(min_size=1, max_size=10, alphabet=DICT_KEY_ALPHABET),
        scalar_values,
        max_size=5,
    )
)
@settings(max_examples=100)
def test_dict_value_roundtrip(mapping):
    """Test that dict values (data, not sections) roundtrip correctly."""
    cfg = Config({"section": {"field": mapping}})
    config_str = cfg.to_str()
    parsed = Config().from_str(config_str)
    assert_values_equal(parsed["section"]["field"], mapping)


@given(
    section1=config_section,
    section2=config_section,
)
@settings(max_examples=100)
def test_multiple_sections_roundtrip(section1, section2):
    """Test that multiple sections roundtrip correctly."""
    cfg = Config({"section1": section1, "section2": section2})
    config_str = cfg.to_str()
    parsed = Config().from_str(config_str)
    assert_values_equal(parsed["section1"], section1)
    assert_values_equal(parsed["section2"], section2)


# =============================================================================
# Nested Sections
# =============================================================================


@given(data=nested_config())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_nested_sections_roundtrip(data):
    """Test that nested sections like [outer.inner] roundtrip correctly."""
    config, path, content = data
    cfg = Config(config)
    config_str = cfg.to_str()
    parsed = Config().from_str(config_str)

    # Navigate to the nested content
    node = parsed
    for name in path:
        node = node[name]

    assert_values_equal(node, content)


# =============================================================================
# Positional Args ([section.*.name] syntax)
# =============================================================================


@given(data=config_with_positional_args())
@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_positional_args_roundtrip(data):
    """Test that [section.*.name] positional args syntax roundtrips correctly."""
    parent_name, positional_names, parent_fields, positional_contents, expected = data

    # Build config string manually since Config() from dict doesn't create the
    # [section.*.name] syntax - it uses the tuple under "*" key
    lines = [f"[{parent_name}]"]
    for key, value in parent_fields.items():
        lines.append(f"{key} = {srsly.json_dumps(value)}")

    for name, content in zip(positional_names, positional_contents):
        lines.append(f"\n[{parent_name}.*.{name}]")
        for key, value in content.items():
            lines.append(f"{key} = {srsly.json_dumps(value)}")

    config_str = "\n".join(lines)

    # Parse and verify
    parsed = Config().from_str(config_str)
    assert_values_equal(dict(parsed), expected)

    # Verify roundtrip
    regenerated = parsed.to_str()
    parsed2 = Config().from_str(regenerated)
    assert_values_equal(dict(parsed2), expected)


# =============================================================================
# Variable Interpolation
# =============================================================================


@given(data=config_with_interpolation())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_variable_interpolation(data):
    """Test that variable interpolation ${section.key} works correctly."""
    config, expected_target = data

    # Create config from dict (uninterpolated)
    cfg = Config(config)

    # Convert to string and back (still uninterpolated)
    config_str = cfg.to_str()
    parsed = Config().from_str(config_str)

    # Interpolate
    interpolated = parsed.interpolate()

    # Check target section has interpolated values
    assert_values_equal(interpolated["target"], expected_target)


# =============================================================================
# Config String Parsing (from INI format)
# =============================================================================


@given(data=config_string())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_config_string_parsing(data):
    """Test that generated config strings parse correctly."""
    config_str, expected = data
    parsed = Config().from_str(config_str)
    assert_values_equal(dict(parsed), expected)


@given(data=config_string())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_config_string_roundtrip(data):
    """Test that config strings survive parse -> to_str -> parse roundtrip."""
    config_str, expected = data

    # Parse the generated string
    parsed1 = Config().from_str(config_str)
    assert_values_equal(dict(parsed1), expected)

    # Convert back to string
    regenerated = parsed1.to_str()

    # Parse again
    parsed2 = Config().from_str(regenerated)
    assert_values_equal(dict(parsed2), expected)


# =============================================================================
# String Parsing Edge Cases
# =============================================================================
# These tests document known issues with string values that resemble JSON
# primitives. When a string contains content that looks like a JSON value
# followed by whitespace, the parser incorrectly converts it.


@pytest.mark.parametrize(
    "value",
    [
        # Strings with whitespace that look like numbers
        "0\n",
        "1\t",
        " 42",
        "42 ",
        # Strings with whitespace that look like booleans
        "true\n",
        "false ",
        # Strings with whitespace that look like null
        "null\n",
    ],
)
def test_string_with_whitespace_stays_string(value):
    """Strings that look like JSON primitives with whitespace stay strings."""
    cfg = Config({"section": {"field": value}})
    config_str = cfg.to_str()
    parsed = Config().from_str(config_str)
    assert parsed["section"]["field"] == value


@pytest.mark.parametrize(
    "value",
    [
        # Positive integers and floats - these work
        "123",
        "3.14",
        "0",
        "0.5",
    ],
)
def test_numeric_string_stays_string(value):
    """Strings that look like positive numbers stay strings.

    These cases work because try_dump_json has special handling to double-quote
    strings that match `value.replace(".", "", 1).isdigit()`.
    """
    cfg = Config({"section": {"field": value}})
    config_str = cfg.to_str()
    parsed = Config().from_str(config_str)
    assert parsed["section"]["field"] == value


@pytest.mark.parametrize(
    "value",
    [
        "-42",
        "-3.14",
    ],
)
def test_negative_numeric_string_stays_string(value):
    """Negative numeric strings stay strings."""
    cfg = Config({"section": {"field": value}})
    config_str = cfg.to_str()
    parsed = Config().from_str(config_str)
    assert parsed["section"]["field"] == value


# =============================================================================
# Registry Resolution Tests
# =============================================================================
# Tests for configs with @registry references that get resolved to values


@st.composite
def config_with_catsie(draw):
    """Generate a config with a catsie.v1 registered function.

    catsie.v1 signature: catsie_v1(evil: StrictBool, cute: bool = True) -> str
    Returns "scratch!" if evil else "meow"
    """
    evil = draw(st.booleans())
    cute = draw(st.booleans())

    config = {
        "cat": {
            "@cats": "catsie.v1",
            "evil": evil,
            "cute": cute,
        }
    }

    expected = "scratch!" if evil else "meow"
    return config, expected


@st.composite
def config_with_catsie_v2(draw):
    """Generate a config with a catsie.v2 registered function.

    catsie.v2 signature: catsie_v2(evil: StrictBool, cute: bool = True, cute_level: int = 1) -> str
    Returns "scratch!" if evil, "meow <3" if cute_level > 2, else "meow"
    """
    evil = draw(st.booleans())
    cute = draw(st.booleans())
    cute_level = draw(st.integers(min_value=0, max_value=5))

    config = {
        "cat": {
            "@cats": "catsie.v2",
            "evil": evil,
            "cute": cute,
            "cute_level": cute_level,
        }
    }

    if evil:
        expected = "scratch!"
    elif cute_level > 2:
        expected = "meow <3"
    else:
        expected = "meow"

    return config, expected


@given(data=config_with_catsie())
@settings(max_examples=50)
def test_registry_resolve_catsie_v1(data):
    """Test that catsie.v1 configs resolve correctly."""
    config, expected = data
    cfg = Config(config)
    resolved = my_registry.resolve(cfg)
    assert resolved["cat"] == expected


@given(data=config_with_catsie_v2())
@settings(max_examples=50)
def test_registry_resolve_catsie_v2(data):
    """Test that catsie.v2 configs resolve correctly."""
    config, expected = data
    cfg = Config(config)
    resolved = my_registry.resolve(cfg)
    assert resolved["cat"] == expected


@given(data=config_with_catsie())
@settings(max_examples=50)
def test_registry_roundtrip_with_resolve(data):
    """Test that configs with registry refs survive roundtrip and resolve."""
    config, expected = data
    cfg = Config(config)

    # Roundtrip through string
    config_str = cfg.to_str()
    parsed = Config().from_str(config_str)

    # Resolve should produce same result
    resolved = my_registry.resolve(parsed)
    assert resolved["cat"] == expected


@st.composite
def multiple_registry_sections(draw):
    """Generate a config with multiple sections containing registered functions.

    Example structure:
        [cat1]
        @cats = "catsie.v1"
        evil = false

        [cat2]
        @cats = "catsie.v2"
        evil = true
        cute_level = 3
    """
    cat1_evil = draw(st.booleans())
    cat2_evil = draw(st.booleans())
    cat2_cute_level = draw(st.integers(min_value=0, max_value=5))

    config = {
        "cat1": {
            "@cats": "catsie.v1",
            "evil": cat1_evil,
        },
        "cat2": {
            "@cats": "catsie.v2",
            "evil": cat2_evil,
            "cute_level": cat2_cute_level,
        },
    }

    cat1_expected = "scratch!" if cat1_evil else "meow"
    if cat2_evil:
        cat2_expected = "scratch!"
    elif cat2_cute_level > 2:
        cat2_expected = "meow <3"
    else:
        cat2_expected = "meow"

    return config, cat1_expected, cat2_expected


@given(data=multiple_registry_sections())
@settings(max_examples=50)
def test_multiple_registry_sections_resolve(data):
    """Test that multiple sections with registered functions resolve correctly."""
    config, cat1_expected, cat2_expected = data
    cfg = Config(config)
    resolved = my_registry.resolve(cfg)
    assert resolved["cat1"] == cat1_expected
    assert resolved["cat2"] == cat2_expected


@given(evil=st.booleans(), cute=st.booleans())
@settings(max_examples=20)
def test_registry_fill_adds_defaults(evil, cute):
    """Test that registry.fill() adds default values."""
    # Config without 'cute' parameter (has default)
    config = {"cat": {"@cats": "catsie.v1", "evil": evil}}
    cfg = Config(config)
    filled = my_registry.fill(cfg)

    # Should have cute with default value
    assert filled["cat"]["cute"] is True
    assert filled["cat"]["evil"] == evil


# =============================================================================
# Schema Inference Tests (make_func_schema / get_func_fields)
# =============================================================================
# Tests for inferring Pydantic schemas from function signatures


def make_test_func_int(x: int) -> int:
    return x


def make_test_func_str(x: str) -> str:
    return x


def make_test_func_bool(x: bool) -> bool:
    return x


def make_test_func_float(x: float) -> float:
    return x


def make_test_func_optional(x: int, y: str = "default") -> str:
    return f"{x} {y}"


def make_test_func_all_optional(x: int = 10, y: str = "default") -> str:
    return f"{x} {y}"


def make_test_func_list(x: list) -> list:
    return x


def make_test_func_typed_list(x: list[int]) -> list:
    return x


class TestMakeFuncSchema:
    """Tests for make_func_schema inferring Pydantic schemas from functions."""

    def test_schema_accepts_correct_int(self):
        """Schema accepts correct int type."""
        schema = make_func_schema(make_test_func_int)
        result = schema.model_validate({"x": 42})
        assert result.x == 42

    def test_schema_rejects_wrong_type_for_int(self):
        """Schema rejects string when int expected."""
        schema = make_func_schema(make_test_func_int)
        with pytest.raises(ValidationError):
            schema.model_validate({"x": "not an int"})

    def test_schema_accepts_correct_str(self):
        """Schema accepts correct str type."""
        schema = make_func_schema(make_test_func_str)
        result = schema.model_validate({"x": "hello"})
        assert result.x == "hello"

    def test_schema_accepts_correct_bool(self):
        """Schema accepts correct bool type."""
        schema = make_func_schema(make_test_func_bool)
        result = schema.model_validate({"x": True})
        assert result.x is True

    def test_schema_accepts_correct_float(self):
        """Schema accepts correct float type."""
        schema = make_func_schema(make_test_func_float)
        result = schema.model_validate({"x": 3.14})
        assert result.x == 3.14

    def test_schema_requires_required_param(self):
        """Schema requires parameters without defaults."""
        schema = make_func_schema(make_test_func_optional)
        with pytest.raises(ValidationError):
            schema.model_validate({"y": "provided"})  # missing x

    def test_schema_uses_default_for_optional(self):
        """Schema uses default value for optional parameters."""
        schema = make_func_schema(make_test_func_optional)
        result = schema.model_validate({"x": 5})
        assert result.x == 5
        assert result.y == "default"

    def test_schema_all_optional_uses_defaults(self):
        """Schema uses defaults when all params are optional."""
        schema = make_func_schema(make_test_func_all_optional)
        result = schema.model_validate({})
        assert result.x == 10
        assert result.y == "default"

    def test_schema_rejects_extra_fields(self):
        """Schema rejects extra fields not in function signature."""
        schema = make_func_schema(make_test_func_int)
        with pytest.raises(ValidationError):
            schema.model_validate({"x": 1, "extra": "not allowed"})

    def test_schema_accepts_list(self):
        """Schema accepts list type."""
        schema = make_func_schema(make_test_func_list)
        result = schema.model_validate({"x": [1, 2, 3]})
        assert result.x == [1, 2, 3]


@given(value=st.integers())
@settings(max_examples=50)
def test_schema_int_property(value):
    """Property test: schema validates any integer."""
    schema = make_func_schema(make_test_func_int)
    result = schema.model_validate({"x": value})
    assert result.x == value


@given(value=st.text(max_size=100))
@settings(max_examples=50)
def test_schema_str_property(value):
    """Property test: schema validates any string."""
    schema = make_func_schema(make_test_func_str)
    result = schema.model_validate({"x": value})
    assert result.x == value


@given(value=st.booleans())
@settings(max_examples=10)
def test_schema_bool_property(value):
    """Property test: schema validates any boolean."""
    schema = make_func_schema(make_test_func_bool)
    result = schema.model_validate({"x": value})
    assert result.x == value


@given(value=st.floats(allow_nan=False, allow_infinity=False))
@settings(max_examples=50)
def test_schema_float_property(value):
    """Property test: schema validates any float."""
    schema = make_func_schema(make_test_func_float)
    result = schema.model_validate({"x": value})
    assert result.x == value


@given(required=st.integers(), optional=st.text(max_size=20) | st.none())
@settings(max_examples=50)
def test_schema_optional_property(required, optional):
    """Property test: schema handles required and optional params."""
    schema = make_func_schema(make_test_func_optional)
    if optional is None:
        result = schema.model_validate({"x": required})
        assert result.y == "default"
    else:
        result = schema.model_validate({"x": required, "y": optional})
        assert result.y == optional
    assert result.x == required


@given(items=st.lists(st.integers(), max_size=10))
@settings(max_examples=50)
def test_schema_list_property(items):
    """Property test: schema validates lists."""
    schema = make_func_schema(make_test_func_list)
    result = schema.model_validate({"x": items})
    assert result.x == items

import json
from hypothesis import strategies as st


# Valid config keys: simple identifiers, no dots or special configparser chars
config_keys = st.from_regex(r"[a-z][a-z0-9_]{0,15}", fullmatch=True)

# Strings safe for config values (no $ to avoid interpolation, no " or \ to
# avoid JSON escaping issues)
config_strings = st.text(
    st.characters(whitelist_categories=("L", "N", "Z"), blacklist_characters='$"\\'),
    min_size=0,
    max_size=20,
)

# Scalar leaf values: str, int, float, bool, None
scalar_leaves = st.one_of(
    config_strings,
    st.integers(min_value=-(2**31), max_value=2**31),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
    st.none(),
)

# JSON leaf values: scalars, or lists/dicts of scalars (arbitrarily nested).
# These are values that get serialized as a single JSON-encoded string in the
# config, rather than expanded into subsections.
_json_values = st.recursive(
    scalar_leaves,
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(config_keys, children, max_size=5),
    ),
    max_leaves=10,
)
# Only the non-scalar cases are interesting as "json leaves" — scalars are
# already covered by scalar_leaves.
json_leaves = st.one_of(
    st.lists(scalar_leaves, min_size=0, max_size=5),
    st.dictionaries(config_keys, _json_values, min_size=0, max_size=5),
)

# Basic config leaves (no JSON-encoded collections)
config_leaves = scalar_leaves

# A config node is either a leaf or a dict of config nodes.
config_nodes = st.recursive(
    config_leaves,
    lambda children: st.dictionaries(config_keys, children, min_size=0, max_size=5),
    max_leaves=30,
)

# A valid config must have sections at the top level (all values must be dicts).
config_dicts = st.dictionaries(
    config_keys,
    st.dictionaries(config_keys, config_nodes, min_size=0, max_size=5),
    min_size=1,
    max_size=5,
)


def _leaf_to_str(value):
    """Serialize a leaf value the way confection's config format expects."""
    if value is None:
        return "null"
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, str):
        return json.dumps(value)
    elif isinstance(value, float):
        return json.dumps(value)
    elif isinstance(value, int):
        return str(value)
    elif isinstance(value, (list, dict)):
        return json.dumps(value)
    raise TypeError(f"Unexpected leaf type: {type(value)}")


def _flatten_sections(data, prefix="", inline_paths=frozenset()):
    """Convert a nested dict into a list of (section_path, {key: leaf_str}) pairs.

    Dicts at paths in inline_paths are serialized as inline JSON values rather
    than expanded into subsections.
    """
    sections = []
    leaves = {}
    for key, value in data.items():
        key_path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and key_path not in inline_paths:
            sections.extend(_flatten_sections(value, key_path, inline_paths))
        else:
            leaves[key] = _leaf_to_str(value)
    if prefix:
        sections.insert(0, (prefix, leaves))
    return sections


def serialize_with_inline(data, inline_paths=frozenset()):
    """Serialize a nested dict to a config string, inlining dicts at the given paths."""
    sections = _flatten_sections(data, inline_paths=inline_paths)
    parts = []
    for section_name, leaves in sections:
        parts.append(f"[{section_name}]")
        for key, value_str in leaves.items():
            parts.append(f"{key} = {value_str}")
        parts.append("")
    return "\n".join(parts).strip()


@st.composite
def json_config_dicts(draw):
    """Strategy that produces (data, inline_paths) pairs.

    The data is a nested dict suitable for Config. inline_paths is a set of
    dotted paths where the value is a JSON-encoded leaf (list or dict) rather
    than a config subsection. This distinction matters for serialization: the
    library always expands dicts into subsections, but a valid config string
    could also have them as inline JSON values.
    """
    # Start with a basic config dict (sections with scalar leaves)
    base = draw(config_dicts)
    inline_paths = set()

    # Sprinkle some JSON leaf values into the config
    for section_key, section in list(base.items()):
        for key in list(section.keys()):
            if draw(st.booleans()):
                # Replace some leaves with JSON-encoded values
                value = draw(json_leaves)
                section[key] = value
                inline_paths.add(f"{section_key}.{key}")

    return base, inline_paths

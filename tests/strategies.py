import json
from hypothesis import strategies as st


# Valid config keys: simple identifiers, no dots or special configparser chars
config_keys = st.from_regex(r"[a-z][a-z0-9_]{0,15}", fullmatch=True)

# Leaf values: str, int, float, None
config_leaves = st.one_of(
    st.text(
        st.characters(whitelist_categories=("L", "N", "Z"), blacklist_characters='$"\\'),
        min_size=0,
        max_size=20,
    ),
    st.integers(min_value=-(2**31), max_value=2**31),
    st.floats(allow_nan=False, allow_infinity=False),
    st.none(),
)


def _leaf_to_str(value):
    """Serialize a leaf value the way confection's config format expects."""
    if value is None:
        return "null"
    elif isinstance(value, str):
        return json.dumps(value)
    elif isinstance(value, float):
        return json.dumps(value)
    elif isinstance(value, int):
        return str(value)
    raise TypeError(f"Unexpected leaf type: {type(value)}")


def _flatten_sections(data, prefix=""):
    """Convert a nested dict into a list of (section_path, {key: leaf_str}) pairs."""
    sections = []
    leaves = {}
    for key, value in data.items():
        if isinstance(value, dict):
            child_prefix = f"{prefix}.{key}" if prefix else key
            sections.extend(_flatten_sections(value, child_prefix))
        else:
            leaves[key] = _leaf_to_str(value)
    # Emit this section if it has leaves OR if it's a named section with no children
    # (empty sections are valid in confection)
    if prefix:
        sections.insert(0, (prefix, leaves))
    return sections


def _to_config_str(data):
    """Serialize a nested dict to a confection config string."""
    sections = _flatten_sections(data)
    parts = []
    for section_name, leaves in sections:
        parts.append(f"[{section_name}]")
        for key, value_str in leaves.items():
            parts.append(f"{key} = {value_str}")
        parts.append("")
    return "\n".join(parts).strip()


# A config node is either a leaf or a dict of config nodes.
# We use st.recursive to build arbitrarily nested structures.
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


@st.composite
def config_pairs(draw):
    """Strategy that produces (config_str, expected_dict) pairs."""
    data = draw(config_dicts)
    config_str = _to_config_str(data)
    return config_str, data

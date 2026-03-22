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

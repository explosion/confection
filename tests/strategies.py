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
    lambda children: st.dictionaries(config_keys, children, min_size=0, max_size=4),
    max_leaves=15,
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


def _collect_scalar_paths(data, prefix=""):
    """Collect all (dotted_path, value) pairs for scalar leaves in a config dict."""
    paths = []
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            paths.extend(_collect_scalar_paths(value, path))
        elif not isinstance(value, (list, dict)):
            paths.append((path, value))
    return paths


def _set_at_path(data, path, value):
    """Set a value at a dotted path in a nested dict."""
    parts = path.split(".")
    node = data
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def _deep_copy(obj):
    """Simple deep copy for nested dicts/lists of scalars."""
    if isinstance(obj, dict):
        return {k: _deep_copy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_deep_copy(v) for v in obj]
    return obj


def _make_interpolated_config_str(sections, replacements):
    """Build a config string from sections, applying variable replacements."""
    modified = {name: dict(leaves) for name, leaves in sections}
    for replace_path, target_path in replacements.items():
        section, key = replace_path.rsplit(".", 1)
        if section in modified:
            modified[section][key] = "${" + target_path + "}"
    parts = []
    for section_name, _ in sections:
        parts.append(f"[{section_name}]")
        for key, val in modified.get(section_name, {}).items():
            parts.append(f"{key} = {val}")
        parts.append("")
    return "\n".join(parts).strip()


@st.composite
def interpolated_config(draw):
    """Strategy producing (config_str, expected_dict) with variable interpolation.

    Replaces some scalar leaves with ${section.key} references. Avoids cycles
    by only referencing values that are never themselves replaced.
    """
    base = draw(config_dicts)
    scalar_paths = _collect_scalar_paths(base)
    # Filter out bools — they serialize as true/false which can't be
    # referenced via ${} (configparser treats them specially)
    scalar_paths = [(p, v) for p, v in scalar_paths if not isinstance(v, bool)]
    if len(scalar_paths) < 2:
        return serialize_with_inline(base), base

    sections = _flatten_sections(base)
    expected = _deep_copy(base)

    # Split paths into targets (stable values to reference) and candidates
    # (values that may be replaced with refs). A path can't be both.
    # Use random subset selection instead of permutations (which is O(n!)).
    target_flags = draw(
        st.lists(
            st.booleans(),
            min_size=len(scalar_paths),
            max_size=len(scalar_paths),
        )
    )
    targets = [sp for sp, flag in zip(scalar_paths, target_flags) if flag]
    candidates = [sp for sp, flag in zip(scalar_paths, target_flags) if not flag]

    if not targets or not candidates:
        return serialize_with_inline(base), base

    replacements = {}
    for cand_path, _ in candidates:
        if draw(st.booleans()):
            target_path, target_value = draw(st.sampled_from(targets))
            replacements[cand_path] = target_path
            _set_at_path(expected, cand_path, target_value)

    config_str = _make_interpolated_config_str(sections, replacements)
    return config_str, expected


@st.composite
def circular_interpolated_config(draw):
    """Strategy producing config strings with circular variable references."""
    base = draw(config_dicts)
    scalar_paths = _collect_scalar_paths(base)
    scalar_paths = [(p, v) for p, v in scalar_paths if not isinstance(v, bool)]
    if len(scalar_paths) < 2:
        from hypothesis import assume

        assume(False)

    sections = _flatten_sections(base)

    # Pick 2+ paths and create a cycle: a -> b -> ... -> a
    cycle_len = draw(st.integers(min_value=2, max_value=min(4, len(scalar_paths))))
    cycle_indices = draw(
        st.lists(
            st.sampled_from(range(len(scalar_paths))),
            min_size=cycle_len,
            max_size=cycle_len,
            unique=True,
        )
    )
    cycle_paths = [scalar_paths[i][0] for i in cycle_indices]

    replacements = {}
    for i in range(cycle_len):
        replacements[cycle_paths[i]] = cycle_paths[(i + 1) % cycle_len]

    return _make_interpolated_config_str(sections, replacements)

import io
from configparser import (
    ConfigParser,
    InterpolationMissingOptionError,
    ParsingError
)
from typing import Any, Dict, List, Tuple
from .util import is_promise, try_dump_json, try_load_json, VARIABLE_RE
from ._interpolation import CustomInterpolation
from ._constants import SECTION_PREFIX
from ._errors import ConfigValidationError


def parse_config(
    text: str,
    *,
    interpolate: bool = True,
    overrides: Dict[str, Any] = {},
) -> dict[str, Any]:
    """Parse a config string into a nested dict.

    Handles the full pipeline: parse with ConfigParser, validate structure,
    apply overrides, interpret values, and resolve section references.

    Returns the nested dict and whether a second interpolation pass is needed
    (when overrides were applied with interpolation enabled).
    """
    config_parser = _get_configparser(interpolate=interpolate and not overrides)
    try:
        config_parser.read_string(text)
    except ParsingError as e:
        desc = f"Make sure the sections and values are formatted correctly.\n\n{e}"
        raise ConfigValidationError(desc=desc) from None
    errors = _validate_configparser(config_parser)
    if errors:
        raise errors[0]
    errors = _validate_overrides(config_parser, overrides)
    if errors:
        raise errors[0]
    # Assumes overrides have been pre-validated.
    for key, value in overrides.items():
        section, option = key.rsplit(".", 1)
        config_parser.set(section, option, try_dump_json(value, overrides))
    result: dict[str, Any] = {}
    section_parts = [section.split(".") for section in config_parser.sections()]
    # Build the skeleton of nested dicts from section names.
    for parts in section_parts:
        node = result
        for part in parts[:-1]:
            if part == "*":
                node.setdefault(part, {})
            else:
                node = node[part]
        node.setdefault(parts[-1], {})
    # Fill in values, processing breadth-first by section depth.
    for section, values in sorted(config_parser.items(), key=lambda x: len(x[0].split("."))):
        if section == "DEFAULT":
            continue
        parts = section.split(".")
        node = result
        for part in parts:
            node = node[part]
        for key in values:
            node[key] = _interpret_value(config_parser.get(section, key))
    # Replace section reference placeholders with actual dicts.
    _replace_section_refs(result, result)
    return result


def serialize_config(data: dict[str, Any], *, interpolate: bool = True) -> str:
    """Serialize a nested config dict to a config string."""
    flattened = _get_configparser(interpolate=interpolate)
    queue: list[tuple[tuple, dict[str, Any]]] = [(tuple(), data)]
    for path, node in queue:
        section_name = ".".join(path)
        is_kwarg = path and path[-1] != "*"
        if is_kwarg and not flattened.has_section(section_name):
            flattened.add_section(section_name)
        for key, value in node.items():
            if hasattr(value, "items"):
                # Reference to a function with no arguments, serialize
                # inline as a dict and don't create new section
                if is_promise(value) and len(value) == 1 and is_kwarg:
                    flattened.set(section_name, key, try_dump_json(value, node))
                else:
                    queue.append((path + (key,), value))
            else:
                flattened.set(section_name, key, try_dump_json(value, node))
    string_io = io.StringIO()
    flattened.write(string_io)
    return string_io.getvalue().strip()


def _get_configparser(interpolate: bool = True) -> ConfigParser:
    config = ConfigParser(interpolation=CustomInterpolation() if interpolate else None)
    # Preserve case of keys: https://stackoverflow.com/a/1611877/6400719
    config.optionxform = str  # type: ignore
    return config


def _validate_configparser(config_parser: ConfigParser) -> list[ConfigValidationError]:
    """Validate a configparser's structure before interpreting it into a Config.

    Checks that:
    - No values leak into the DEFAULT section (top-level values without a section)
    - All parent sections exist for dotted section names (e.g. "a.b" requires "a")
    - No key in a section conflicts with a child section name
    - No interpolation errors in values
    """
    errors = []
    default_section = config_parser.defaults()
    if default_section:
        err_title = "Found config values without a top-level section"
        err_msg = "not part of a section"
        err = [{"loc": [k], "msg": err_msg} for k in default_section]
        errors.append(ConfigValidationError(errors=err, title=err_title))
    section_names = set(config_parser.sections())
    for section in config_parser.sections():
        path = section.split(".")
        for i in range(1, len(path)):
            parent = ".".join(path[:i])
            if parent not in section_names:
                err_title = (
                    "Error parsing config section. Perhaps a section name is wrong?"
                )
                err = [{"loc": path, "msg": f"Section '{path[i-1]}' is not defined"}]
                errors.append(ConfigValidationError(errors=err, title=err_title))
                break
        try:
            keys = set(config_parser.options(section))
        except InterpolationMissingOptionError as e:
            errors.append(ConfigValidationError(desc=f"{e}"))
            continue
        for other in section_names:
            if other.startswith(section + "."):
                child = other[len(section) + 1:].split(".")[0]
                if child in keys:
                    err = [{"loc": other.split("."), "msg": "found conflicting values"}]
                    errors.append(ConfigValidationError(errors=err))
    return errors


def _validate_overrides(config_parser: ConfigParser, overrides: dict[str, Any]) -> list[ConfigValidationError]:
    errors = []
    err_title = "Error parsing config overrides"
    for key in overrides:
        err_msg = "not a section value that can be overridden"
        err = [{"loc": key.split("."), "msg": err_msg}]
        if "." not in key:
            errors.append(ConfigValidationError(errors=err, title=err_title))
            continue
        section, _ = key.rsplit(".", 1)
        # Check for section and accept if option not in config[section]
        if section not in config_parser:
            errors.append(ConfigValidationError(errors=err, title=err_title))
        # TODO: Are we supposed to chek for the *option*?
    return errors



def _interpret_value(value: Any) -> Any:
    """Interpret a single config value."""
    result = try_load_json(value)
    # If value is a string and it contains a variable, use original value
    # (not interpreted string, which could lead to double quotes:
    # ${x.y} -> "${x.y}" -> "'${x.y}'"). Make sure to check it's a string,
    # so we're not keeping lists as strings.
    # NOTE: This currently can't handle uninterpolated values like [${x.y}]!
    if isinstance(result, str) and VARIABLE_RE.search(value):
        result = value
    return result


def _replace_section_refs(root: dict[str, Any], node: dict[str, Any], parent: str = "") -> None:
    """Replace section reference placeholders with actual dicts."""
    for key, value in node.items():
        key_parent = f"{parent}.{key}".strip(".")
        if isinstance(value, dict):
            _replace_section_refs(root, value, parent=key_parent)
        elif isinstance(value, list):
            node[key] = [
                _get_section_ref(root, v, parent=[parent, key]) for v in value
            ]
        else:
            node[key] = _get_section_ref(root, value, parent=[parent, key])


def _get_section_ref(root: Dict[str, Any], value: Any, *, parent: List[str] = []) -> Any:
    """Resolve a single section reference placeholder, or return value as-is."""
    if isinstance(value, str) and value.startswith(
        f'"{SECTION_PREFIX}'
    ):  # pragma: no cover
        value = try_load_json(value)  # pragma: no cover
    if (
        isinstance(value, str)
        and value.startswith(SECTION_PREFIX)
        and value != SECTION_PREFIX
    ):
        parts = value.replace(SECTION_PREFIX, "", 1).split(".")
        result = root
        for item in parts:
            result = result[item]
        return result
    elif (
        isinstance(value, str)
        and SECTION_PREFIX in value
        and value != SECTION_PREFIX
    ):
        err_desc = (
            "Can't reference whole sections or return values of function "
            "blocks inside a string or list\n\nYou can change your variable to "
            "reference a value instead. Keep in mind that it's not "
            "possible to interpolate the return value of a registered "
            "function, since variables are interpolated when the config "
            "is loaded, and registered functions are resolved afterwards."
        )
        err = [{"loc": parent, "msg": "uses section variable in string or list"}]
        raise ConfigValidationError(errors=err, desc=err_desc)
    return value


__all__ = ["parse_config", "serialize_config"]

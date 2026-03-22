from configparser import (
    ConfigParser,
    InterpolationMissingOptionError,
    ParsingError
)
from typing import Any
from .util import try_dump_json
from ._interpolation import CustomInterpolation
from ._errors import ConfigValidationError


def get_configparser(interpolate: bool = True) -> ConfigParser:
    config = ConfigParser(interpolation=CustomInterpolation() if interpolate else None)
    # Preserve case of keys: https://stackoverflow.com/a/1611877/6400719
    config.optionxform = str  # type: ignore
    return config


def validate_configparser(config_parser: ConfigParser) -> list[ConfigValidationError]:
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


def validate_overrides(config_parser: ConfigParser, overrides: dict[str, Any]) -> list[ConfigValidationError]:
    errors = []
    err_title = "Error parsing config overrides"
    for key in overrides:
        err_msg = "not a section value that can be overridden"
        err = [{"loc": key.split("."), "msg": err_msg}]
        if "." not in key:
            errors.append(ConfigValidationError(errors=err, title=err_title))
        section, _ = key.rsplit(".", 1)
        # Check for section and accept if option not in config[section]
        if section not in config_parser:
            errors.append(ConfigValidationError(errors=err, title=err_title))
        # TODO: Are we supposed to chek for the *option*?
    return errors


def set_overrides(config: ConfigParser, overrides: dict[str, Any]) -> None:
    """Set overrides in the ConfigParser before config is interpreted."""
    # Assumes overrides have been pre-validated.
    for key, value in overrides.items():
        section, option = key.rsplit(".", 1)
        config.set(section, option, try_dump_json(value, overrides))


__all__ = ["ConfigParser", "get_configparser", "validate_configparser", "validate_overrides", "set_overrides", "ParsingError"]

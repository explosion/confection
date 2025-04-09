import copy
import io
import re
import warnings
from configparser import (
    MAX_INTERPOLATION_DEPTH,
    ConfigParser,
    ExtendedInterpolation,
    InterpolationDepthError,
    InterpolationMissingOptionError,
    InterpolationSyntaxError,
    NoOptionError,
    NoSectionError,
    ParsingError,
)
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import srsly
from ._errors import ConfigValidationError
from .util import is_promise


# Field used for positional arguments, e.g. [section.*.xyz]. The alias is
# required for the schema (shouldn't clash with user-defined arg names)
ARGS_FIELD = "*"
ARGS_FIELD_ALIAS = "VARIABLE_POSITIONAL_ARGS"
# Aliases for fields that would otherwise shadow pydantic attributes. Can be any
# string, so we're using name + space so it looks the same in error messages etc.
RESERVED_FIELDS = {"validate": "validate\u0020"}
# Internal prefix used to mark section references for custom interpolation
SECTION_PREFIX = "__SECTION__:"
# Values that shouldn't be loaded during interpolation because it'd cause
# even explicit string values to be incorrectly parsed as bools/None etc.
JSON_EXCEPTIONS = ("true", "false", "null")
# Regex to detect whether a value contains a variable
VARIABLE_RE = re.compile(r"\$\{[\w\.:]+\}")


class CustomInterpolation(ExtendedInterpolation):
    def before_read(self, parser, section, option, value):
        # If we're dealing with a quoted string as the interpolation value,
        # make sure we load and unquote it so we don't end up with '"value"'
        try:
            json_value = srsly.json_loads(value)
            if isinstance(json_value, str) and json_value not in JSON_EXCEPTIONS:
                value = json_value
        except ValueError:
            if value and value[0] == value[-1] == "'":
                warnings.warn(
                    f"The value [{value}] seems to be single-quoted, but values "
                    "use JSON formatting, which requires double quotes."
                )
        except Exception:
            pass
        return super().before_read(parser, section, option, value)

    def before_get(self, parser, section, option, value, defaults):
        # Mostly copy-pasted from the built-in configparser implementation.
        L = []
        self.interpolate(parser, option, L, value, section, defaults, 1)
        return "".join(L)

    def interpolate(self, parser, option, accum, rest, section, map, depth):
        # Mostly copy-pasted from the built-in configparser implementation.
        # We need to overwrite this method so we can add special handling for
        # block references :( All values produced here should be strings –
        # we need to wait until the whole config is interpreted anyways so
        # filling in incomplete values here is pointless. All we need is the
        # section reference so we can fetch it later.
        rawval = parser.get(section, option, raw=True, fallback=rest)
        if depth > MAX_INTERPOLATION_DEPTH:
            raise InterpolationDepthError(option, section, rawval)
        while rest:
            p = rest.find("$")
            if p < 0:
                accum.append(rest)
                return
            if p > 0:
                accum.append(rest[:p])
                rest = rest[p:]
            # p is no longer used
            c = rest[1:2]
            if c == "$":
                accum.append("$")
                rest = rest[2:]
            elif c == "{":
                # We want to treat both ${a:b} and ${a.b} the same
                m = self._KEYCRE.match(rest)  # type: ignore
                if m is None:
                    err = f"bad interpolation variable reference {rest}"
                    raise InterpolationSyntaxError(option, section, err)
                orig_var = m.group(1)
                path = orig_var.replace(":", ".").rsplit(".", 1)
                rest = rest[m.end() :]
                sect = section
                opt = option
                try:
                    if len(path) == 1:
                        opt = parser.optionxform(path[0])
                        if opt in map:
                            v = map[opt]
                        else:
                            # We have block reference, store it as a special key
                            section_name = parser[parser.optionxform(path[0])]._name
                            v = self._get_section_name(section_name)
                    elif len(path) == 2:
                        sect = path[0]
                        opt = parser.optionxform(path[1])
                        fallback = "__FALLBACK__"
                        v = parser.get(sect, opt, raw=True, fallback=fallback)
                        # If a variable doesn't exist, try again and treat the
                        # reference as a section
                        if v == fallback:
                            v = self._get_section_name(parser[f"{sect}.{opt}"]._name)
                    else:
                        err = f"More than one ':' found: {rest}"
                        raise InterpolationSyntaxError(option, section, err)
                except (KeyError, NoSectionError, NoOptionError):
                    raise InterpolationMissingOptionError(
                        option, section, rawval, orig_var
                    ) from None
                if "$" in v:
                    new_map = dict(parser.items(sect, raw=True))
                    self.interpolate(parser, opt, accum, v, sect, new_map, depth + 1)
                else:
                    accum.append(v)
            else:
                err = "'$' must be followed by '$' or '{', found: %r" % (rest,)
                raise InterpolationSyntaxError(option, section, err)

    def _get_section_name(self, name: str) -> str:
        """Generate the name of a section. Note that we use a quoted string here
        so we can use section references within lists and load the list as
        JSON. Since section references can't be used within strings, we don't
        need the quoted vs. unquoted distinction like we do for variables.

        Examples (assuming section = {"foo": 1}):
            - value: ${section.foo} -> value: 1
            - value: "hello ${section.foo}" -> value: "hello 1"
            - value: ${section} -> value: {"foo": 1}
            - value: "${section}" -> value: {"foo": 1}
            - value: "hello ${section}" -> invalid
        """
        return f'"{SECTION_PREFIX}{name}"'


def get_configparser(interpolate: bool = True):
    config = ConfigParser(interpolation=CustomInterpolation() if interpolate else None)
    # Preserve case of keys: https://stackoverflow.com/a/1611877/6400719
    config.optionxform = str  # type: ignore
    return config


class Config(dict):
    """This class holds the model and training configuration and can load and
    save the TOML-style configuration format from/to a string, file or bytes.
    The Config class is a subclass of dict and uses Python's ConfigParser
    under the hood.
    """

    is_interpolated: bool

    def __init__(
        self,
        data: Optional[Union[Dict[str, Any], "ConfigParser", "Config"]] = None,
        *,
        is_interpolated: Optional[bool] = None,
        section_order: Optional[List[str]] = None,
    ) -> None:
        """Initialize a new Config object with optional data."""
        dict.__init__(self)
        if data is None:
            data = {}
        if not isinstance(data, (dict, Config, ConfigParser)):
            raise ValueError(
                f"Can't initialize Config with data. Expected dict, Config or "
                f"ConfigParser but got: {type(data)}"
            )
        # Whether the config has been interpolated. We can use this to check
        # whether we need to interpolate again when it's resolved. We assume
        # that a config is interpolated by default.
        if is_interpolated is not None:
            self.is_interpolated = is_interpolated
        elif isinstance(data, Config):
            self.is_interpolated = data.is_interpolated
        else:
            self.is_interpolated = True
        if section_order is not None:
            self.section_order = section_order
        elif isinstance(data, Config):
            self.section_order = data.section_order
        else:
            self.section_order = []
        # Update with data
        self.update(self._sort(data))

    def interpolate(self) -> "Config":
        """Interpolate a config. Returns a copy of the object."""
        # This is currently the most effective way because we need our custom
        # to_str logic to run in order to re-serialize the values so we can
        # interpolate them again. ConfigParser.read_dict will just call str()
        # on all values, which isn't enough.
        return Config().from_str(self.to_str())

    def interpret_config(self, config: "ConfigParser") -> None:
        """Interpret a config, parse nested sections and parse the values
        as JSON. Mostly used internally and modifies the config in place.
        """
        self._validate_sections(config)
        # Sort sections by depth, so that we can iterate breadth-first. This
        # allows us to check that we're not expanding an undefined block.
        get_depth = lambda item: len(item[0].split("."))
        for section, values in sorted(config.items(), key=get_depth):
            if section == "DEFAULT":
                # Skip [DEFAULT] section so it doesn't cause validation error
                continue
            parts = section.split(".")
            node = self
            for part in parts[:-1]:
                if part == "*":
                    node = node.setdefault(part, {})
                elif part not in node:
                    err_title = (
                        "Error parsing config section. Perhaps a section name is wrong?"
                    )
                    err = [{"loc": parts, "msg": f"Section '{part}' is not defined"}]
                    raise ConfigValidationError(
                        config=self, errors=err, title=err_title
                    )
                else:
                    node = node[part]
            if not isinstance(node, dict):
                # Happens if both value *and* subsection were defined for a key
                err = [{"loc": parts, "msg": "found conflicting values"}]
                err_cfg = f"{self}\n{({part: dict(values)})}"
                raise ConfigValidationError(config=err_cfg, errors=err)
            # Set the default section
            node = node.setdefault(parts[-1], {})
            if not isinstance(node, dict):
                # Happens if both value *and* subsection were defined for a key
                err = [{"loc": parts, "msg": "found conflicting values"}]
                err_cfg = f"{self}\n{({part: dict(values)})}"
                raise ConfigValidationError(config=err_cfg, errors=err)
            try:
                keys_values = list(values.items())
            except InterpolationMissingOptionError as e:
                raise ConfigValidationError(desc=f"{e}") from None
            for key, value in keys_values:
                config_v = config.get(section, key)
                node[key] = self._interpret_value(config_v)
        self.replace_section_refs(self)

    def replace_section_refs(
        self, config: Union[Dict[str, Any], "Config"], parent: str = ""
    ) -> None:
        """Replace references to section blocks in the final config."""
        for key, value in config.items():
            key_parent = f"{parent}.{key}".strip(".")
            if isinstance(value, dict):
                self.replace_section_refs(value, parent=key_parent)
            elif isinstance(value, list):
                config[key] = [
                    self._get_section_ref(v, parent=[parent, key]) for v in value
                ]
            else:
                config[key] = self._get_section_ref(value, parent=[parent, key])

    def _interpret_value(self, value: Any) -> Any:
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

    def _get_section_ref(self, value: Any, *, parent: List[str] = []) -> Any:
        """Get a single section reference."""
        if isinstance(value, str) and value.startswith(f'"{SECTION_PREFIX}'):
            value = try_load_json(value)
        if isinstance(value, str) and value.startswith(SECTION_PREFIX):
            parts = value.replace(SECTION_PREFIX, "").split(".")
            result = self
            for item in parts:
                try:
                    result = result[item]
                except (KeyError, TypeError):  # This should never happen
                    err_title = "Error parsing reference to config section"
                    err_msg = f"Section '{'.'.join(parts)}' is not defined"
                    err = [{"loc": parts, "msg": err_msg}]
                    raise ConfigValidationError(
                        config=self, errors=err, title=err_title
                    ) from None
            return result
        elif isinstance(value, str) and SECTION_PREFIX in value:
            # String value references a section (either a dict or return
            # value of promise). We can't allow this, since variables are
            # always interpolated *before* configs are resolved.
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

    def copy(self) -> "Config":
        """Deepcopy the config."""
        try:
            config = copy.deepcopy(self)
        except Exception as e:
            raise ValueError(f"Couldn't deep-copy config: {e}") from e
        return Config(
            config,
            is_interpolated=self.is_interpolated,
            section_order=self.section_order,
        )

    def merge(
        self, updates: Union[Dict[str, Any], "Config"], remove_extra: bool = False
    ) -> "Config":
        """Deep merge the config with updates, using current as defaults."""
        defaults = self.copy()
        updates = Config(updates).copy()
        merged = deep_merge_configs(updates, defaults, remove_extra=remove_extra)
        return Config(
            merged,
            is_interpolated=defaults.is_interpolated and updates.is_interpolated,
            section_order=defaults.section_order,
        )

    def _sort(
        self, data: Union["Config", "ConfigParser", Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Sort sections using the currently defined sort order. Sort
        sections by index on section order, if available, then alphabetic, and
        account for subsections, which should always follow their parent.
        """
        sort_map = {section: i for i, section in enumerate(self.section_order)}
        sort_key = lambda x: (
            sort_map.get(x[0].split(".")[0], len(sort_map)),
            _mask_positional_args(x[0]),
        )
        return dict(sorted(data.items(), key=sort_key))

    def _set_overrides(self, config: "ConfigParser", overrides: Dict[str, Any]) -> None:
        """Set overrides in the ConfigParser before config is interpreted."""
        err_title = "Error parsing config overrides"
        for key, value in overrides.items():
            err_msg = "not a section value that can be overridden"
            err = [{"loc": key.split("."), "msg": err_msg}]
            if "." not in key:
                raise ConfigValidationError(errors=err, title=err_title)
            section, option = key.rsplit(".", 1)
            # Check for section and accept if option not in config[section]
            if section not in config:
                raise ConfigValidationError(errors=err, title=err_title)
            config.set(section, option, try_dump_json(value, overrides))

    def _validate_sections(self, config: "ConfigParser") -> None:
        # If the config defines top-level properties that are not sections (e.g.
        # if config was constructed from dict), those values would be added as
        # [DEFAULTS] and included in *every other section*. This is usually not
        # what we want and it can lead to very confusing results.
        default_section = config.defaults()
        if default_section:
            err_title = "Found config values without a top-level section"
            err_msg = "not part of a section"
            err = [{"loc": [k], "msg": err_msg} for k in default_section]
            raise ConfigValidationError(errors=err, title=err_title)

    def from_str(
        self, text: str, *, interpolate: bool = True, overrides: Dict[str, Any] = {}
    ) -> "Config":
        """Load the config from a string."""
        config = get_configparser(interpolate=interpolate)
        if overrides:
            config = get_configparser(interpolate=False)
        try:
            config.read_string(text)
        except ParsingError as e:
            desc = f"Make sure the sections and values are formatted correctly.\n\n{e}"
            raise ConfigValidationError(desc=desc) from None
        config._sections = self._sort(config._sections)
        self._set_overrides(config, overrides)
        self.clear()
        self.interpret_config(config)
        if overrides and interpolate:
            # do the interpolation. Avoids recursion because the new call from_str call will have overrides as empty
            self = self.interpolate()
        self.is_interpolated = interpolate
        return self

    def to_str(self, *, interpolate: bool = True) -> str:
        """Write the config to a string."""
        flattened = get_configparser(interpolate=interpolate)
        queue: List[Tuple[tuple, "Config"]] = [(tuple(), self)]
        for path, node in queue:
            section_name = ".".join(path)
            is_kwarg = path and path[-1] != "*"
            if is_kwarg and not flattened.has_section(section_name):
                # Always create sections for non-'*' sections, not only if
                # they have leaf entries, as we don't want to expand
                # blocks that are undefined
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
        # Order so subsection follow parent (not all sections, then all subs etc.)
        flattened._sections = self._sort(flattened._sections)
        self._validate_sections(flattened)
        string_io = io.StringIO()
        flattened.write(string_io)
        return string_io.getvalue().strip()

    def to_bytes(self, *, interpolate: bool = True) -> bytes:
        """Serialize the config to a byte string."""
        return self.to_str(interpolate=interpolate).encode("utf8")

    def from_bytes(
        self,
        bytes_data: bytes,
        *,
        interpolate: bool = True,
        overrides: Dict[str, Any] = {},
    ) -> "Config":
        """Load the config from a byte string."""
        return self.from_str(
            bytes_data.decode("utf8"), interpolate=interpolate, overrides=overrides
        )

    def to_disk(self, path: Union[str, Path], *, interpolate: bool = True):
        """Serialize the config to a file."""
        path = Path(path) if isinstance(path, str) else path
        with path.open("w", encoding="utf8") as file_:
            file_.write(self.to_str(interpolate=interpolate))

    def from_disk(
        self,
        path: Union[str, Path],
        *,
        interpolate: bool = True,
        overrides: Dict[str, Any] = {},
    ) -> "Config":
        """Load config from a file."""
        path = Path(path) if isinstance(path, str) else path
        with path.open("r", encoding="utf8") as file_:
            text = file_.read()
        return self.from_str(text, interpolate=interpolate, overrides=overrides)


def _mask_positional_args(name: str) -> List[Optional[str]]:
    """Create a section name representation that masks names
    of positional arguments to retain their order in sorts."""

    stable_name = cast(List[Optional[str]], name.split("."))

    # Remove names of sections that are a positional argument.
    for i in range(1, len(stable_name)):
        if stable_name[i - 1] == "*":
            stable_name[i] = None

    return stable_name


def try_load_json(value: str) -> Any:
    """Load a JSON string if possible, otherwise default to original value."""
    try:
        return srsly.json_loads(value)
    except Exception:
        return value


def try_dump_json(value: Any, data: Union[Dict[str, dict], Config, str] = "") -> str:
    """Dump a config value as JSON and output user-friendly error if it fails."""
    # Special case if we have a variable: it's already a string so don't dump
    # to preserve ${x:y} vs. "${x:y}"
    if isinstance(value, str) and VARIABLE_RE.search(value):
        return value
    if isinstance(value, str) and value.replace(".", "", 1).isdigit():
        # Work around values that are strings but numbers
        value = f'"{value}"'
    try:
        value = srsly.json_dumps(value)
        value = re.sub(r"\$([^{])", "$$\1", value)
        value = re.sub(r"\$$", "$$", value)
        return value
    except Exception as e:
        err_msg = (
            f"Couldn't serialize config value of type {type(value)}: {e}. Make "
            f"sure all values in your config are JSON-serializable. If you want "
            f"to include Python objects, use a registered function that returns "
            f"the object instead."
        )
        raise ConfigValidationError(config=data, desc=err_msg) from e


def deep_merge_configs(
    config: Union[Dict[str, Any], Config],
    defaults: Union[Dict[str, Any], Config],
    *,
    remove_extra: bool = False,
) -> Union[Dict[str, Any], Config]:
    """Deep merge two configs."""
    if remove_extra:
        # Filter out values in the original config that are not in defaults
        keys = list(config.keys())
        for key in keys:
            if key not in defaults:
                del config[key]
    for key, value in defaults.items():
        if isinstance(value, dict):
            node = config.setdefault(key, {})
            if not isinstance(node, dict):
                continue
            value_promises = [k for k in value if k.startswith("@")]
            value_promise = value_promises[0] if value_promises else None
            node_promises = [k for k in node if k.startswith("@")] if node else []
            node_promise = node_promises[0] if node_promises else None
            # We only update the block from defaults if it refers to the same
            # registered function
            if (
                value_promise
                and node_promise
                and (
                    value_promise in node
                    and node[value_promise] != value[value_promise]
                )
            ):
                continue
            if node_promise and (
                node_promise not in value or node[node_promise] != value[node_promise]
            ):
                continue
            defaults = deep_merge_configs(node, value, remove_extra=remove_extra)
        elif key not in config:
            config[key] = value
    return config

import copy
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Self

from ._errors import ConfigValidationError, ConfectionError
from .util import is_promise, try_dump_json
from ._parser import get_configparser, ConfigParser, interpret_configparser, validate_configparser, validate_overrides, ParsingError, set_overrides


class Config(dict):
    # TODO: Improve doc string
    """Dict subclass to save TOML-style configuration format from/to string, file
    or bytes.
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
            raise ConfectionError(
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
        self.update(data)

    def interpolate(self) -> Self:
        """Interpolate (resolve var references) a config.

        Returns a copy of the object.
        """
        # This is currently the most effective way because we need our custom
        # to_str logic to run in order to re-serialize the values so we can
        # interpolate them again. ConfigParser.read_dict will just call str()
        # on all values, which isn't enough.
        return type(self)().from_str(self.to_str())

    def copy(self) -> Self:
        """Deepcopy the config."""
        try:
            config = copy.deepcopy(self)
        except Exception as e:
            raise ConfectionError(f"Couldn't deep-copy config: {e}") from e
        return type(self)(
            config,
            is_interpolated=self.is_interpolated,
            section_order=self.section_order,
        )

    def merge(
        self, updates: Union[Dict[str, Any], "Config"], remove_extra: bool = False
    ) -> Self:
        """Deep merge the config with updates, using current as defaults."""
        defaults = self.copy()
        updates = Config(updates).copy()
        merged = deep_merge_configs(updates, defaults, remove_extra=remove_extra)
        return type(self)(
            merged,
            is_interpolated=defaults.is_interpolated and updates.is_interpolated,
            section_order=defaults.section_order,
        )

    def from_str(
        self, text: str, *, interpolate: bool = True, overrides: Dict[str, Any] = {}
    ) -> Self:
        """Load the config from a string."""
        config_parser = get_configparser(interpolate=interpolate and not overrides)
        try:
            config_parser.read_string(text)
        except ParsingError as e:
            desc = f"Make sure the sections and values are formatted correctly.\n\n{e}"
            raise ConfigValidationError(desc=desc) from None
        errors = validate_configparser(config_parser)
        if errors:
            raise errors[0]
        errors = validate_overrides(config_parser, overrides)
        if errors:
            raise errors[0]
        set_overrides(config_parser, overrides)
        # Clear previous values and populate from the configparser
        self.clear()
        self.update(interpret_configparser(config_parser))
        if overrides and interpolate:
            # do the interpolation. Avoids recursion because the new call from_str call
            # will have overrides as empty
            self = self.interpolate()
        # TODO: How does this make sense? If we had no overrides but interpolate=False,
        # shouldn't we set is_interpolated=True?
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
    ) -> Self:
        """Load the config from a byte string."""
        return self.from_str(
            bytes_data.decode("utf8"), interpolate=interpolate, overrides=overrides
        )

    def to_disk(self, path: Union[str, Path], *, interpolate: bool = True) -> None:
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
    ) -> Self:
        """Load config from a file."""
        path = Path(path) if isinstance(path, str) else path
        with path.open("r", encoding="utf8") as file_:
            text = file_.read()
        return self.from_str(text, interpolate=interpolate, overrides=overrides)


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



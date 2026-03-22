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
)
from typing import Dict, List

from ._constants import SECTION_PREFIX


class CustomInterpolation(ExtendedInterpolation):
    def before_read(
        self, parser: ConfigParser, section: str, option: str, value: str
    ) -> str:
        # Warn about single-quoted strings (common mistake)
        if value and value[0] == value[-1] == "'":
            warnings.warn(
                f"The value [{value}] seems to be single-quoted, but values "
                "use JSON formatting, which requires double quotes."
            )
        return super().before_read(parser, section, option, value)

    def _coerce_for_string_context(self, v: str) -> str:
        """Coerce a raw config value for use in a compound string expression."""
        import json

        # Don't coerce section references - they need to stay quoted for JSON
        if SECTION_PREFIX in v:
            return v
        try:
            parsed = json.loads(v)
        except json.JSONDecodeError:
            return v  # Not valid JSON, already a plain string
        if isinstance(parsed, str):
            return parsed  # Unwrap JSON string
        # Use json.dumps() for non-strings, escaping inner quotes so they don't
        # conflict with the outer JSON string quotes
        return json.dumps(parsed).replace('"', '\\"')

    def before_get(
        self,
        parser: ConfigParser,
        section: str,
        option: str,
        value: str,
        defaults: Dict[str, str],
    ) -> str:
        # Mostly copy-pasted from the built-in configparser implementation.
        # The interpolate() method resolves ${...} references and appends pieces
        # to L. For a bare reference like ${x}, L has one element. For compound
        # expressions like "hello ${x}", L has multiple pieces that we join.
        # Compound results stay as strings (coerced via _coerce_for_string_context),
        # while bare references keep their JSON type for _interpret_value to parse.
        L: List[str] = []
        self.interpolate(parser, option, L, value, section, defaults, 1)
        if len(L) == 1:
            return L[0]
        return "".join(self._coerce_for_string_context(piece) for piece in L)

    def interpolate(
        self,
        parser: ConfigParser,
        option: str,
        accum: List[str],
        rest: str,
        section: str,
        map: Dict[str, str],
        depth: int,
    ) -> None:
        """Resolve variable references like ${foo.bar}"""
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
                m = self._KEYCRE.match(rest)  # type: ignore[attr-defined]
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
                            section_name = parser[parser.optionxform(path[0])]._name  # type: ignore[union-attr]
                            v = self._get_section_name(section_name)
                    elif len(path) == 2:
                        sect = path[0]
                        opt = parser.optionxform(path[1])
                        fallback = "__FALLBACK__"
                        v = parser.get(sect, opt, raw=True, fallback=fallback)
                        # If a variable doesn't exist, try again and treat the
                        # reference as a section
                        if v == fallback:
                            v = self._get_section_name(parser[f"{sect}.{opt}"]._name)  # type: ignore[union-attr]
                    else:  # pragma: no cover
                        # Dead code: rsplit(".", 1) produces at most 2 elements
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




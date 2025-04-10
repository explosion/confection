import inspect
import pickle
import platform
from types import GeneratorType
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    Sequence,
    Literal,
)

import catalogue
import pytest

from pydantic import BaseModel, StrictFloat, PositiveInt, constr
from pydantic.types import StrictBool

from confection import Config, ConfigValidationError
from confection.tests.util import Cat, make_tempdir, my_registry
from confection.util import Generator, partial

EXAMPLE_CONFIG = """
[optimizer]
@optimizers = "Adam.v1"
beta1 = 0.9
beta2 = 0.999
use_averages = true

[optimizer.learn_rate]
@schedules = "warmup_linear.v1"
initial_rate = 0.1
warmup_steps = 10000
total_steps = 100000

[pipeline]

[pipeline.classifier]
name = "classifier"
factory = "classifier"

[pipeline.classifier.model]
@layers = "ClassifierModel.v1"
hidden_depth = 1
hidden_width = 64
token_vector_width = 128

[pipeline.classifier.model.embedding]
@layers = "Embedding.v1"
width = ${pipeline.classifier.model:token_vector_width}

"""

OPTIMIZER_CFG = """
[optimizer]
@optimizers = "Adam.v1"
beta1 = 0.9
beta2 = 0.999
use_averages = true

[optimizer.learn_rate]
@schedules = "warmup_linear.v1"
initial_rate = 0.1
warmup_steps = 10000
total_steps = 100000
"""


class HelloIntsSchema(BaseModel):
    hello: int
    world: int
    model_config = {"extra": "forbid"}


class DefaultsSchema(BaseModel):
    required: int
    optional: str = "default value"
    model_config = {"extra": "forbid"}


class LooseSchema(BaseModel):
    required: int
    optional: str = "default value"
    model_config = {"extra": "allow"}


class ComplexSchema(BaseModel):
    outer_req: int
    outer_opt: str = "default value"

    level2_req: HelloIntsSchema
    level2_opt: DefaultsSchema = DefaultsSchema(required=1)


good_catsie = {"@cats": "catsie.v1", "evil": False, "cute": True}
ok_catsie = {"@cats": "catsie.v1", "evil": False, "cute": False}
bad_catsie = {"@cats": "catsie.v1", "evil": True, "cute": True}
worst_catsie = {"@cats": "catsie.v1", "evil": True, "cute": False}


def test_read_config():
    byte_string = EXAMPLE_CONFIG.encode("utf8")
    cfg = Config().from_bytes(byte_string)

    assert cfg["optimizer"]["beta1"] == 0.9
    assert cfg["optimizer"]["learn_rate"]["initial_rate"] == 0.1
    assert cfg["pipeline"]["classifier"]["factory"] == "classifier"
    assert cfg["pipeline"]["classifier"]["model"]["embedding"]["width"] == 128


@pytest.mark.skip
def test_optimizer_config():
    cfg = Config().from_str(OPTIMIZER_CFG)
    optimizer = my_registry.resolve(cfg, validate=True)["optimizer"]
    assert optimizer.beta1 == 0.9


def test_config_to_str():
    cfg = Config().from_str(OPTIMIZER_CFG)
    assert cfg.to_str().strip() == OPTIMIZER_CFG.strip()
    cfg = Config({"optimizer": {"foo": "bar"}}).from_str(OPTIMIZER_CFG)
    assert cfg.to_str().strip() == OPTIMIZER_CFG.strip()


def test_config_to_str_creates_intermediate_blocks():
    cfg = Config({"optimizer": {"foo": {"bar": 1}}})
    assert (
        cfg.to_str().strip()
        == """
[optimizer]

[optimizer.foo]
bar = 1
    """.strip()
    )


def test_config_to_str_escapes():
    section_str = """
        [section]
        node1 = "^a$$"
        node2 = "$$b$$c"
        """
    section_dict = {"section": {"node1": "^a$", "node2": "$b$c"}}

    # parse from escaped string
    cfg = Config().from_str(section_str)
    assert cfg == section_dict

    # parse from non-escaped dict
    cfg = Config(section_dict)
    assert cfg == section_dict

    # roundtrip through str
    cfg_str = cfg.to_str()
    assert "^a$$" in cfg_str
    new_cfg = Config().from_str(cfg_str)
    assert cfg == section_dict


def test_config_roundtrip_bytes():
    cfg = Config().from_str(OPTIMIZER_CFG)
    cfg_bytes = cfg.to_bytes()
    new_cfg = Config().from_bytes(cfg_bytes)
    assert new_cfg.to_str().strip() == OPTIMIZER_CFG.strip()


def test_config_roundtrip_disk():
    cfg = Config().from_str(OPTIMIZER_CFG)
    with make_tempdir() as path:
        cfg_path = path / "config.cfg"
        cfg.to_disk(cfg_path)
        new_cfg = Config().from_disk(cfg_path)
    assert new_cfg.to_str().strip() == OPTIMIZER_CFG.strip()


def test_config_roundtrip_disk_respects_path_subclasses(pathy_fixture):
    cfg = Config().from_str(OPTIMIZER_CFG)
    cfg_path = pathy_fixture / "config.cfg"
    cfg.to_disk(cfg_path)
    new_cfg = Config().from_disk(cfg_path)
    assert new_cfg.to_str().strip() == OPTIMIZER_CFG.strip()


def test_config_to_str_invalid_defaults():
    """Test that an error is raised if a config contains top-level keys without
    a section that would otherwise be interpreted as [DEFAULT] (which causes
    the values to be included in *all* other sections).
    """
    cfg = {"one": 1, "two": {"@cats": "catsie.v1", "evil": "hello"}}
    with pytest.raises(ConfigValidationError):
        Config(cfg).to_str()
    config_str = "[DEFAULT]\none = 1"
    with pytest.raises(ConfigValidationError):
        Config().from_str(config_str)


def test_validation_custom_types():
    def complex_args(
        rate: StrictFloat,
        steps: PositiveInt = 10,  # type: ignore
        log_level: Literal["ERROR", "INFO"] = "ERROR",  # noqa: F821
    ):
        return None

    my_registry.complex = catalogue.create(
        my_registry.namespace, "complex", entry_points=False
    )
    my_registry.complex("complex.v1")(complex_args)
    cfg = {"@complex": "complex.v1", "rate": 1.0, "steps": 20, "log_level": "INFO"}
    my_registry.resolve({"config": cfg})
    cfg = {"@complex": "complex.v1", "rate": 1.0, "steps": -1, "log_level": "INFO"}
    with pytest.raises(ConfigValidationError):
        # steps is not a positive int
        my_registry.resolve({"config": cfg})
    cfg = {"@complex": "complex.v1", "rate": 1.0, "steps": 20, "log_level": "none"}
    with pytest.raises(ConfigValidationError):
        # log_level is not a string matching the regex
        my_registry.resolve({"config": cfg})
    cfg = {"@complex": "complex.v1", "rate": 1.0, "steps": 20, "log_level": "INFO"}
    with pytest.raises(ConfigValidationError):
        # top-level object is promise
        my_registry.resolve(cfg)
    with pytest.raises(ConfigValidationError):
        # top-level object is promise
        my_registry.fill(cfg)
    cfg = {"@complex": "complex.v1", "rate": 1.0, "@cats": "catsie.v1"}
    with pytest.raises(ConfigValidationError):
        # two constructors
        my_registry.resolve({"config": cfg})


@my_registry.cats("catsie.v666")
def catsie_666(*args, meow=False):
    return args


@my_registry.cats("var_args_optional.v1")
def cats_var_args_optional(*args: str, foo: str = "hi"):
    return " ".join(args) + f"foo={foo}"


@my_registry.cats("catsie.v777")
def catsie_777(y: int = 1):
    return "meow" * y


@pytest.mark.parametrize(
    "cfg",
    [
        """[a]\nb = 1\n* = ["foo","bar"]""",
        """[a]\nb = 1\n\n[a.*.bar]\ntest = 2\n\n[a.*.foo]\ntest = 1""",
    ],
)
def test_positional_args_round_trip(cfg: str):
    round_trip = Config().from_str(cfg).to_str()
    assert round_trip == cfg


@pytest.mark.parametrize(
    "cfg,expected",
    [
        (
            """[a]\n@cats = "catsie.v666"\n\n[a.*.foo]\n@cats = "catsie.v777\"""",
            """[a]\n@cats = "catsie.v666"\nmeow = false\n\n[a.*.foo]\n@cats = "catsie.v777"\ny = 1""",
        ),
        (
            """[a]\n@cats = "var_args_optional.v1"\n* = ["meow","bar"]""",
            """[a]\n@cats = "var_args_optional.v1"\n* = ["meow","bar"]\nfoo = \"hi\"""",
        ),
        (
            """[a]\n@cats = "catsie.v666"\n\n[a.*.foo]\nx = 1""",
            """[a]\n@cats = "catsie.v666"\nmeow = false\n\n[a.*.foo]\nx = 1""",
        ),
        (
            """[a]\n@cats = "catsie.v666"\n\n[a.*.foo]\n@cats = "catsie.v777\"""",
            """[a]\n@cats = "catsie.v666"\nmeow = false\n\n[a.*.foo]\n@cats = "catsie.v777"\ny = 1""",
        ),
    ],
)
def test_positional_args_fill_round_trip(cfg, expected):
    config = Config().from_str(cfg)
    filled_dict = my_registry.fill(config)
    filled = filled_dict.to_str()
    assert filled == expected


@pytest.mark.parametrize(
    "cfg,expected",
    [
        (
            """[a]\nb = 1\n\n[a.*.bar]\ntest = 2\n\n[a.*.foo]\ntest = 1""",
            {"a": {"*": ({"test": 2}, {"test": 1}), "b": 1}},
        ),
        ("""[a]\n@cats = "catsie.v666"\n\n[a.*.foo]\nx = 1""", {"a": ({"x": 1},)}),
        (
            """[a]\n@cats = "catsie.v666"\n\n[a.*.foo]\n@cats = "catsie.v777"\ny = 3""",
            {"a": ("meowmeowmeow",)},
        ),
    ],
)
def test_positional_args_resolve_round_trip(cfg, expected):
    resolved = my_registry.resolve(Config().from_str(cfg))
    assert resolved == expected


@pytest.mark.parametrize(
    "cfg",
    [
        "[a]\nb = 1\nc = 2\n\n[a.c]\nd = 3",
        "[a]\nb = 1\n\n[a.c]\nd = 2\n\n[a.c.d]\ne = 3",
    ],
)
def test_handle_error_duplicate_keys(cfg):
    """This would cause very cryptic error when interpreting config.
    (TypeError: 'X' object does not support item assignment)
    """
    with pytest.raises(ConfigValidationError):
        Config().from_str(cfg)


@pytest.mark.parametrize(
    "cfg,is_valid",
    [("[a]\nb = 1\n\n[a.c]\nd = 3", True), ("[a]\nb = 1\n\n[A.c]\nd = 2", False)],
)
def test_cant_expand_undefined_block(cfg, is_valid):
    """Test that you can't expand a block that hasn't been created yet. This
    comes up when you typo a name, and if we allow expansion of undefined blocks,
    it's very hard to create good errors for those typos.
    """
    if is_valid:
        Config().from_str(cfg)
    else:
        with pytest.raises(ConfigValidationError):
            Config().from_str(cfg)


def test_resolve_prefilled_values():
    class Language(object):
        def __init__(self):
            ...

    @my_registry.optimizers("prefilled.v1")
    def prefilled(nlp: Language, value: int = 10):
        return (nlp, value)

    # Passing an instance of Language here via the config is bad, since it
    # won't serialize to a string, but we still test for it
    config = {"test": {"@optimizers": "prefilled.v1", "nlp": Language(), "value": 50}}
    resolved = my_registry.resolve(config, validate=True)
    result = resolved["test"]
    assert isinstance(result[0], Language)
    assert result[1] == 50


def test_deepcopy_config():
    config = Config({"a": 1, "b": {"c": 2, "d": 3}})
    copied = config.copy()
    # Same values but not same object
    assert config == copied
    assert config is not copied


@pytest.mark.skipif(
    platform.python_implementation() == "PyPy", reason="copy does not fail for pypy"
)
def test_deepcopy_config_pickle():
    numpy = pytest.importorskip("numpy")
    # Check for error if value can't be pickled/deepcopied
    config = Config({"a": 1, "b": numpy})
    with pytest.raises(ValueError):
        config.copy()


def test_config_to_str_simple_promises():
    """Test that references to function registries without arguments are
    serialized inline as dict."""
    config_str = """[section]\nsubsection = {"@registry":"value"}"""
    config = Config().from_str(config_str)
    assert config["section"]["subsection"]["@registry"] == "value"
    assert config.to_str() == config_str


def test_config_from_str_invalid_section():
    config_str = """[a]\nb = null\n\n[a.b]\nc = 1"""
    with pytest.raises(ConfigValidationError):
        Config().from_str(config_str)

    config_str = """[a]\nb = null\n\n[a.b.c]\nd = 1"""
    with pytest.raises(ConfigValidationError):
        Config().from_str(config_str)


def test_config_to_str_order():
    """Test that Config.to_str orders the sections."""
    config = {"a": {"b": {"c": 1, "d": 2}, "e": 3}, "f": {"g": {"h": {"i": 4, "j": 5}}}}
    expected = (
        "[a]\ne = 3\n\n[a.b]\nc = 1\nd = 2\n\n[f]\n\n[f.g]\n\n[f.g.h]\ni = 4\nj = 5"
    )
    config = Config(config)
    assert config.to_str() == expected


@pytest.mark.parametrize("d", [".", ":"])
def test_config_interpolation(d):
    """Test that config values are interpolated correctly. The parametrized
    value is the final divider (${a.b} vs. ${a:b}). Both should now work and be
    valid. The double {{ }} in the config strings are required to prevent the
    references from being interpreted as an actual f-string variable.
    """
    c_str = """[a]\nfoo = "hello"\n\n[b]\nbar = ${foo}"""
    with pytest.raises(ConfigValidationError):
        Config().from_str(c_str)
    c_str = f"""[a]\nfoo = "hello"\n\n[b]\nbar = ${{a{d}foo}}"""
    assert Config().from_str(c_str)["b"]["bar"] == "hello"
    c_str = f"""[a]\nfoo = "hello"\n\n[b]\nbar = ${{a{d}foo}}!"""
    assert Config().from_str(c_str)["b"]["bar"] == "hello!"
    c_str = f"""[a]\nfoo = "hello"\n\n[b]\nbar = "${{a{d}foo}}!\""""
    assert Config().from_str(c_str)["b"]["bar"] == "hello!"
    c_str = f"""[a]\nfoo = 15\n\n[b]\nbar = ${{a{d}foo}}!"""
    assert Config().from_str(c_str)["b"]["bar"] == "15!"
    c_str = f"""[a]\nfoo = ["x", "y"]\n\n[b]\nbar = ${{a{d}foo}}"""
    assert Config().from_str(c_str)["b"]["bar"] == ["x", "y"]
    # Interpolation within the same section
    c_str = f"""[a]\nfoo = "x"\nbar = ${{a{d}foo}}\nbaz = "${{a{d}foo}}y\""""
    assert Config().from_str(c_str)["a"]["bar"] == "x"
    assert Config().from_str(c_str)["a"]["baz"] == "xy"


def test_config_interpolation_lists():
    # Test that lists are preserved correctly
    c_str = """[a]\nb = 1\n\n[c]\nd = ["hello ${a.b}", "world"]"""
    config = Config().from_str(c_str, interpolate=False)
    assert config["c"]["d"] == ["hello ${a.b}", "world"]
    config = config.interpolate()
    assert config["c"]["d"] == ["hello 1", "world"]
    c_str = """[a]\nb = 1\n\n[c]\nd = [${a.b}, "hello ${a.b}", "world"]"""
    config = Config().from_str(c_str)
    assert config["c"]["d"] == [1, "hello 1", "world"]
    config = Config().from_str(c_str, interpolate=False)
    # NOTE: This currently doesn't work, because we can't know how to JSON-load
    # the uninterpolated list [${a.b}].
    # assert config["c"]["d"] == ["${a.b}", "hello ${a.b}", "world"]
    # config = config.interpolate()
    # assert config["c"]["d"] == [1, "hello 1", "world"]
    c_str = """[a]\nb = 1\n\n[c]\nd = ["hello", ${a}]"""
    config = Config().from_str(c_str)
    assert config["c"]["d"] == ["hello", {"b": 1}]
    c_str = """[a]\nb = 1\n\n[c]\nd = ["hello", "hello ${a}"]"""
    with pytest.raises(ConfigValidationError):
        Config().from_str(c_str)
    config_str = """[a]\nb = 1\n\n[c]\nd = ["hello", {"x": ["hello ${a.b}"], "y": 2}]"""
    config = Config().from_str(config_str)
    assert config["c"]["d"] == ["hello", {"x": ["hello 1"], "y": 2}]
    config_str = """[a]\nb = 1\n\n[c]\nd = ["hello", {"x": [${a.b}], "y": 2}]"""
    with pytest.raises(ConfigValidationError):
        Config().from_str(c_str)


@pytest.mark.parametrize("d", [".", ":"])
def test_config_interpolation_sections(d):
    """Test that config sections are interpolated correctly. The parametrized
    value is the final divider (${a.b} vs. ${a:b}). Both should now work and be
    valid. The double {{ }} in the config strings are required to prevent the
    references from being interpreted as an actual f-string variable.
    """
    # Simple block references
    c_str = """[a]\nfoo = "hello"\nbar = "world"\n\n[b]\nc = ${a}"""
    config = Config().from_str(c_str)
    assert config["b"]["c"] == config["a"]
    # References with non-string values
    c_str = f"""[a]\nfoo = "hello"\n\n[a.x]\ny = ${{a{d}b}}\n\n[a.b]\nc = 1\nd = [10]"""
    config = Config().from_str(c_str)
    assert config["a"]["x"]["y"] == config["a"]["b"]
    # Multiple references in the same string
    c_str = f"""[a]\nx = "string"\ny = 10\n\n[b]\nz = "${{a{d}x}}/${{a{d}y}}\""""
    config = Config().from_str(c_str)
    assert config["b"]["z"] == "string/10"
    # Non-string references in string (converted to string)
    c_str = f"""[a]\nx = ["hello", "world"]\n\n[b]\ny = "result: ${{a{d}x}}\""""
    config = Config().from_str(c_str)
    assert config["b"]["y"] == 'result: ["hello", "world"]'
    # References to sections referencing sections
    c_str = """[a]\nfoo = "x"\n\n[b]\nbar = ${a}\n\n[c]\nbaz = ${b}"""
    config = Config().from_str(c_str)
    assert config["b"]["bar"] == config["a"]
    assert config["c"]["baz"] == config["b"]
    # References to section values referencing other sections
    c_str = f"""[a]\nfoo = "x"\n\n[b]\nbar = ${{a}}\n\n[c]\nbaz = ${{b{d}bar}}"""
    config = Config().from_str(c_str)
    assert config["c"]["baz"] == config["b"]["bar"]
    # References to sections with subsections
    c_str = """[a]\nfoo = "x"\n\n[a.b]\nbar = 100\n\n[c]\nbaz = ${a}"""
    config = Config().from_str(c_str)
    assert config["c"]["baz"] == config["a"]
    # Infinite recursion
    c_str = """[a]\nfoo ="x"\n\n[a.b]\nbar = ${a}"""
    config = Config().from_str(c_str)
    assert config["a"]["b"]["bar"] == config["a"]
    c_str = f"""[a]\nfoo = "x"\n\n[b]\nbar = ${{a}}\n\n[c]\nbaz = ${{b.bar{d}foo}}"""
    # We can't reference not-yet interpolated subsections
    with pytest.raises(ConfigValidationError):
        Config().from_str(c_str)
    # Generally invalid references
    c_str = f"""[a]\nfoo = ${{b{d}bar}}"""
    with pytest.raises(ConfigValidationError):
        Config().from_str(c_str)
    # We can't reference sections or promises within strings
    c_str = """[a]\n\n[a.b]\nfoo = "x: ${c}"\n\n[c]\nbar = 1\nbaz = 2"""
    with pytest.raises(ConfigValidationError):
        Config().from_str(c_str)


def test_config_from_str_overrides():
    config_str = """[a]\nb = 1\n\n[a.c]\nd = 2\ne = 3\n\n[f]\ng = {"x": "y"}"""
    # Basic value substitution
    overrides = {"a.b": 10, "a.c.d": 20}
    config = Config().from_str(config_str, overrides=overrides)
    assert config["a"]["b"] == 10
    assert config["a"]["c"]["d"] == 20
    assert config["a"]["c"]["e"] == 3
    # Valid values that previously weren't in config
    config = Config().from_str(config_str, overrides={"a.c.f": 100})
    assert config["a"]["c"]["d"] == 2
    assert config["a"]["c"]["e"] == 3
    assert config["a"]["c"]["f"] == 100
    # Invalid keys and sections
    with pytest.raises(ConfigValidationError):
        Config().from_str(config_str, overrides={"f": 10})
    # This currently isn't expected to work, because the dict in f.g is not
    # interpreted as a section while the config is still just the configparser
    with pytest.raises(ConfigValidationError):
        Config().from_str(config_str, overrides={"f.g.x": "z"})
    # With variables (values)
    config_str = """[a]\nb = 1\n\n[a.c]\nd = 2\ne = ${a:b}"""
    config = Config().from_str(config_str, overrides={"a.b": 10})
    assert config["a"]["b"] == 10
    assert config["a"]["c"]["e"] == 10
    # With variables (sections)
    config_str = """[a]\nb = 1\n\n[a.c]\nd = 2\n[e]\nf = ${a.c}"""
    config = Config().from_str(config_str, overrides={"a.c.d": 20})
    assert config["a"]["c"]["d"] == 20
    assert config["e"]["f"] == {"d": 20}


@pytest.mark.parametrize("d", [".", ":"])
def test_config_no_interpolation(d):
    """Test that interpolation is correctly preserved. The parametrized
    value is the final divider (${a.b} vs. ${a:b}). Both should now work and be
    valid. The double {{ }} in the config strings are required to prevent the
    references from being interpreted as an actual f-string variable.
    """
    numpy = pytest.importorskip("numpy")
    c_str = f"""[a]\nb = 1\n\n[c]\nd = ${{a{d}b}}\ne = \"hello${{a{d}b}}"\nf = ${{a}}"""
    config = Config().from_str(c_str, interpolate=False)
    assert not config.is_interpolated
    assert config["c"]["d"] == f"${{a{d}b}}"
    assert config["c"]["e"] == f'"hello${{a{d}b}}"'
    assert config["c"]["f"] == "${a}"
    config2 = Config().from_str(config.to_str(), interpolate=True)
    assert config2.is_interpolated
    assert config2["c"]["d"] == 1
    assert config2["c"]["e"] == "hello1"
    assert config2["c"]["f"] == {"b": 1}
    config3 = config.interpolate()
    assert config3.is_interpolated
    assert config3["c"]["d"] == 1
    assert config3["c"]["e"] == "hello1"
    assert config3["c"]["f"] == {"b": 1}
    # Bad non-serializable value
    cfg = {"x": {"y": numpy.asarray([[1, 2], [4, 5]], dtype="f"), "z": f"${{x{d}y}}"}}
    with pytest.raises(ConfigValidationError):
        Config(cfg).interpolate()


def test_config_no_interpolation_registry():
    config_str = """[a]\nbad = true\n[b]\n@cats = "catsie.v1"\nevil = ${a:bad}\n\n[c]\n d = ${b}"""
    config = Config().from_str(config_str, interpolate=False)
    assert not config.is_interpolated
    assert config["b"]["evil"] == "${a:bad}"
    assert config["c"]["d"] == "${b}"
    filled = my_registry.fill(config)
    resolved = my_registry.resolve(config)
    assert resolved["b"] == "scratch!"
    assert resolved["c"]["d"] == "scratch!"
    assert filled["b"]["evil"] == "${a:bad}"
    assert filled["b"]["cute"] is True
    assert filled["c"]["d"] == "${b}"
    interpolated = filled.interpolate()
    assert interpolated.is_interpolated
    assert interpolated["b"]["evil"] is True
    assert interpolated["c"]["d"] == interpolated["b"]
    config = Config().from_str(config_str, interpolate=True)
    assert config.is_interpolated
    filled = my_registry.fill(config)
    resolved = my_registry.resolve(config)
    assert resolved["b"] == "scratch!"
    assert resolved["c"]["d"] == "scratch!"
    assert filled["b"]["evil"] is True
    assert filled["c"]["d"] == filled["b"]
    # Resolving a non-interpolated filled config
    config = Config().from_str(config_str, interpolate=False)
    assert not config.is_interpolated
    filled = my_registry.fill(config)
    assert not filled.is_interpolated
    assert filled["c"]["d"] == "${b}"
    resolved = my_registry.resolve(filled)
    assert resolved["c"]["d"] == "scratch!"


def test_config_deep_merge():
    config = {"a": "hello", "b": {"c": "d"}}
    defaults = {"a": "world", "b": {"c": "e", "f": "g"}}
    merged = Config(defaults).merge(config)
    assert len(merged) == 2
    assert merged["a"] == "hello"
    assert merged["b"] == {"c": "d", "f": "g"}
    config = {"a": "hello", "b": {"@test": "x", "foo": 1}}
    defaults = {"a": "world", "b": {"@test": "x", "foo": 100, "bar": 2}, "c": 100}
    merged = Config(defaults).merge(config)
    assert len(merged) == 3
    assert merged["a"] == "hello"
    assert merged["b"] == {"@test": "x", "foo": 1, "bar": 2}
    assert merged["c"] == 100
    config = {"a": "hello", "b": {"@test": "x", "foo": 1}, "c": 100}
    defaults = {"a": "world", "b": {"@test": "y", "foo": 100, "bar": 2}}
    merged = Config(defaults).merge(config)
    assert len(merged) == 3
    assert merged["a"] == "hello"
    assert merged["b"] == {"@test": "x", "foo": 1}
    assert merged["c"] == 100
    # Test that leaving out the factory just adds to existing
    config = {"a": "hello", "b": {"foo": 1}, "c": 100}
    defaults = {"a": "world", "b": {"@test": "y", "foo": 100, "bar": 2}}
    merged = Config(defaults).merge(config)
    assert len(merged) == 3
    assert merged["a"] == "hello"
    assert merged["b"] == {"@test": "y", "foo": 1, "bar": 2}
    assert merged["c"] == 100
    # Test that switching to a different factory prevents the default from being added
    config = {"a": "hello", "b": {"@foo": 1}, "c": 100}
    defaults = {"a": "world", "b": {"@bar": "y"}}
    merged = Config(defaults).merge(config)
    assert len(merged) == 3
    assert merged["a"] == "hello"
    assert merged["b"] == {"@foo": 1}
    assert merged["c"] == 100
    config = {"a": "hello", "b": {"@foo": 1}, "c": 100}
    defaults = {"a": "world", "b": "y"}
    merged = Config(defaults).merge(config)
    assert len(merged) == 3
    assert merged["a"] == "hello"
    assert merged["b"] == {"@foo": 1}
    assert merged["c"] == 100


def test_config_deep_merge_variables():
    config_str = """[a]\nb= 1\nc = 2\n\n[d]\ne = ${a:b}"""
    defaults_str = """[a]\nx = 100\n\n[d]\ny = 500"""
    config = Config().from_str(config_str, interpolate=False)
    defaults = Config().from_str(defaults_str)
    merged = defaults.merge(config)
    assert merged["a"] == {"b": 1, "c": 2, "x": 100}
    assert merged["d"] == {"e": "${a:b}", "y": 500}
    assert merged.interpolate()["d"] == {"e": 1, "y": 500}
    # With variable in defaults: overwritten by new value
    config = Config().from_str("""[a]\nb= 1\nc = 2""")
    defaults = Config().from_str("""[a]\nb = 100\nc = ${a:b}""", interpolate=False)
    merged = defaults.merge(config)
    assert merged["a"]["c"] == 2


def test_config_to_str_roundtrip():
    numpy = pytest.importorskip("numpy")
    cfg = {"cfg": {"foo": False}}
    config_str = Config(cfg).to_str()
    assert config_str == "[cfg]\nfoo = false"
    config = Config().from_str(config_str)
    assert dict(config) == cfg
    cfg = {"cfg": {"foo": "false"}}
    config_str = Config(cfg).to_str()
    assert config_str == '[cfg]\nfoo = "false"'
    config = Config().from_str(config_str)
    assert dict(config) == cfg
    # Bad non-serializable value
    cfg = {"cfg": {"x": numpy.asarray([[1, 2, 3, 4], [4, 5, 3, 4]], dtype="f")}}
    config = Config(cfg)
    with pytest.raises(ConfigValidationError):
        config.to_str()
    # Roundtrip with variables: preserve variables correctly (quoted/unquoted)
    config_str = """[a]\nb = 1\n\n[c]\nd = ${a:b}\ne = \"hello${a:b}"\nf = "${a:b}\""""
    config = Config().from_str(config_str, interpolate=False)
    assert config.to_str() == config_str


def test_config_is_interpolated():
    """Test that a config object correctly reports whether it's interpolated."""
    config_str = """[a]\nb = 1\n\n[c]\nd = ${a:b}\ne = \"hello${a:b}"\nf = ${a}"""
    config = Config().from_str(config_str, interpolate=False)
    assert not config.is_interpolated
    config = config.merge(Config({"x": {"y": "z"}}))
    assert not config.is_interpolated
    config = Config(config)
    assert not config.is_interpolated
    config = config.interpolate()
    assert config.is_interpolated
    config = config.merge(Config().from_str(config_str, interpolate=False))
    assert not config.is_interpolated


@pytest.mark.parametrize(
    "section_order,expected_str,expected_keys",
    [
        # fmt: off
        ([], "[a]\nb = 1\nc = 2\n\n[a.d]\ne = 3\n\n[a.f]\ng = 4\n\n[h]\ni = 5\n\n[j]\nk = 6", ["a", "h", "j"]),
        (["j", "h", "a"], "[j]\nk = 6\n\n[h]\ni = 5\n\n[a]\nb = 1\nc = 2\n\n[a.d]\ne = 3\n\n[a.f]\ng = 4", ["j", "h", "a"]),
        (["h"], "[h]\ni = 5\n\n[a]\nb = 1\nc = 2\n\n[a.d]\ne = 3\n\n[a.f]\ng = 4\n\n[j]\nk = 6", ["h", "a", "j"])
        # fmt: on
    ],
)
def test_config_serialize_custom_sort(section_order, expected_str, expected_keys):
    cfg = {
        "j": {"k": 6},
        "a": {"b": 1, "d": {"e": 3}, "c": 2, "f": {"g": 4}},
        "h": {"i": 5},
    }
    cfg_str = Config(cfg).to_str()
    assert Config(cfg, section_order=section_order).to_str() == expected_str
    keys = list(Config(section_order=section_order).from_str(cfg_str).keys())
    assert keys == expected_keys
    keys = list(Config(cfg, section_order=section_order).keys())
    assert keys == expected_keys


def test_config_custom_sort_preserve():
    """Test that sort order is preserved when merging and copying configs,
    or when configs are filled and resolved."""
    cfg = {"x": {}, "y": {}, "z": {}}
    section_order = ["y", "z", "x"]
    expected = "[y]\n\n[z]\n\n[x]"
    config = Config(cfg, section_order=section_order)
    assert config.to_str() == expected
    config2 = config.copy()
    assert config2.to_str() == expected
    config3 = config.merge({"a": {}})
    assert config3.to_str() == f"{expected}\n\n[a]"
    config4 = Config(config)
    assert config4.to_str() == expected
    config_str = """[a]\nb = 1\n[c]\n@cats = "catsie.v1"\nevil = true\n\n[t]\n x = 2"""
    section_order = ["c", "a", "t"]
    config5 = Config(section_order=section_order).from_str(config_str)
    assert list(config5.keys()) == section_order
    filled = my_registry.fill(config5)
    assert filled.section_order == section_order


def test_config_pickle():
    config = Config({"foo": "bar"}, section_order=["foo", "bar", "baz"])
    data = pickle.dumps(config)
    config_new = pickle.loads(data)
    assert config_new == {"foo": "bar"}
    assert config_new.section_order == ["foo", "bar", "baz"]


def test_config_parsing_error():
    config_str = "[a]\nb c"
    with pytest.raises(ConfigValidationError):
        Config().from_str(config_str)


def test_config_dataclasses():
    cat = Cat("testcat", value_in=1, value_out=2)
    config = {"cfg": {"@cats": "catsie.v3", "arg": cat}}
    result = my_registry.resolve(config)["cfg"]
    assert isinstance(result, Cat)
    assert result.name == cat.name
    assert result.value_in == cat.value_in
    assert result.value_out == cat.value_out


@pytest.mark.parametrize(
    "greeting,value,expected",
    [
        # simple substitution should go fine
        [342, "${vars.a}", int],
        ["342", "${vars.a}", str],
        ["everyone", "${vars.a}", str],
    ],
)
def test_config_interpolates(greeting, value, expected):
    str_cfg = f"""
    [project]
    my_par = {value}

    [vars]
    a = "something"
    """
    overrides = {"vars.a": greeting}
    cfg = Config().from_str(str_cfg, overrides=overrides)
    assert type(cfg["project"]["my_par"]) == expected


@pytest.mark.parametrize(
    "greeting,value,expected",
    [
        # fmt: off
        # simple substitution should go fine
        ["hello 342", "${vars.a}", "hello 342"],
        ["hello everyone", "${vars.a}", "hello everyone"],
        ["hello tout le monde", "${vars.a}", "hello tout le monde"],
        ["hello 42", "${vars.a}", "hello 42"],
        # substituting an element in a list
        ["hello 342", "[1, ${vars.a}, 3]", "hello 342"],
        ["hello everyone", "[1, ${vars.a}, 3]", "hello everyone"],
        ["hello tout le monde", "[1, ${vars.a}, 3]", "hello tout le monde"],
        ["hello 42", "[1, ${vars.a}, 3]", "hello 42"],
        # substituting part of a string
        [342, "hello ${vars.a}", "hello 342"],
        ["everyone", "hello ${vars.a}", "hello everyone"],
        ["tout le monde", "hello ${vars.a}", "hello tout le monde"],
        pytest.param("42", "hello ${vars.a}", "hello 42", marks=pytest.mark.xfail),
        # substituting part of a implicit string inside a list
        [342, "[1, hello ${vars.a}, 3]", "hello 342"],
        ["everyone", "[1, hello ${vars.a}, 3]", "hello everyone"],
        ["tout le monde", "[1, hello ${vars.a}, 3]", "hello tout le monde"],
        pytest.param("42", "[1, hello ${vars.a}, 3]", "hello 42", marks=pytest.mark.xfail),
        # substituting part of a explicit string inside a list
        [342, "[1, 'hello ${vars.a}', '3']", "hello 342"],
        ["everyone", "[1, 'hello ${vars.a}', '3']", "hello everyone"],
        ["tout le monde", "[1, 'hello ${vars.a}', '3']", "hello tout le monde"],
        pytest.param("42", "[1, 'hello ${vars.a}', '3']", "hello 42", marks=pytest.mark.xfail),
        # more complicated example
        [342, "[{'name':'x','script':['hello ${vars.a}']}]", "hello 342"],
        ["everyone", "[{'name':'x','script':['hello ${vars.a}']}]", "hello everyone"],
        ["tout le monde", "[{'name':'x','script':['hello ${vars.a}']}]", "hello tout le monde"],
        pytest.param("42", "[{'name':'x','script':['hello ${vars.a}']}]", "hello 42", marks=pytest.mark.xfail),
        # fmt: on
    ],
)
def test_config_overrides(greeting, value, expected):
    str_cfg = f"""
    [project]
    commands = {value}

    [vars]
    a = "world"
    """
    overrides = {"vars.a": greeting}
    assert "${vars.a}" in str_cfg
    cfg = Config().from_str(str_cfg, overrides=overrides)
    assert expected in str(cfg)


def test_warn_single_quotes():
    str_cfg = """
    [project]
    commands = 'do stuff'
    """

    with pytest.warns(UserWarning, match="single-quoted"):
        Config().from_str(str_cfg)

    # should not warn if single quotes are in the middle
    str_cfg = """
    [project]
    commands = some'thing
    """
    Config().from_str(str_cfg)


def test_parse_strings_interpretable_as_ints():
    """Test whether strings interpretable as integers are parsed correctly (i. e. as strings)."""
    cfg = Config().from_str(
        f"""[a]\nfoo = [${{b.bar}}, "00${{b.bar}}", "y"]\n\n[b]\nbar = 3"""  # noqa: F541
    )
    assert cfg["a"]["foo"] == [3, "003", "y"]
    assert cfg["b"]["bar"] == 3

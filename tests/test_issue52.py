"""Test for issue #52: Config with `*` (positional) args should roundtrip
to str and back.
"""

from confection import Config


def test_config_star_args_to_str():
    """Config.to_str() should not crash when sections contain * args."""
    test_cfg = """[test]
foo = "bar"

[test.*]
bar = "foo"
"""
    cfg = Config().from_str(test_cfg)
    # This should not raise NoSectionError
    result = cfg.to_str()
    assert "[test]" in result


def test_config_star_args_roundtrip():
    """Config with * args should survive a from_str -> to_str -> from_str cycle."""
    test_cfg = """[test]
@cats = "catsie.v1"
evil = false

[test.*.foo]
x = 1
"""
    cfg = Config().from_str(test_cfg)
    cfg_str = cfg.to_str()
    cfg2 = Config().from_str(cfg_str)
    assert cfg == cfg2

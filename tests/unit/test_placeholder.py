"""Placeholder test to ensure pytest runs successfully."""


def test_placeholder() -> None:
    """Placeholder test case."""
    assert True


def test_imports() -> None:
    """Test that core workspace packages can be imported."""
    import orchestrator  # noqa: F401
    import rpc  # noqa: F401
    import tts  # noqa: F401

    assert True

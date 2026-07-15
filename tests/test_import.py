"""Test project basics and the public import surface."""

from __future__ import annotations

import leport


def test_public_package_imports() -> None:
    """Core APIs and version metadata are available after a clean import."""

    assert leport.__version__ == "0.1.0"
    assert leport.SOURCE_ADAPTER_API_VERSION == 1
    assert callable(leport.inspect)
    assert callable(leport.create_plan)
    assert callable(leport.convert)
    assert callable(leport.merge)
    assert callable(leport.validate)

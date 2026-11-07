"""Smoke tests for the MachinePulse package."""

import machinepulse


def test_package_version() -> None:
    """The installed package exposes its declared version."""
    assert machinepulse.__version__ == "0.1.0"

"""Shared test configuration.

Registers the ``market_data`` marker used by tests that talk to the Massive
REST API. Those tests are skipped automatically when ``MASSIVE_API_KEY`` is
unset.
"""

from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "market_data: test requires live Massive REST API access (MASSIVE_API_KEY)",
    )


def pytest_collection_modifyitems(config: pytest.Config,
                                  items: list[pytest.Item]) -> None:
    if os.environ.get("MASSIVE_API_KEY"):
        return
    skip = pytest.mark.skip(reason="MASSIVE_API_KEY not set")
    for item in items:
        if "market_data" in item.keywords:
            item.add_marker(skip)

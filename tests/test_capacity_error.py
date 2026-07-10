"""Tests for _is_capacity_error — the detector that maps free-tier LLM 429 /
provider-exhaustion failures to a clear user message instead of a generic error."""

from __future__ import annotations

from ssu_agent import main


def test_detects_all_providers_exhausted() -> None:
    assert main._is_capacity_error(RuntimeError("All LLM providers exhausted")) is True


def test_detects_wrapped_rate_limit_error() -> None:
    class RateLimitError(Exception):
        status_code = 429

    try:
        try:
            raise RateLimitError("Error code: 429 - temporarily rate-limited upstream")
        except RateLimitError as inner:
            raise RuntimeError("All LLM providers exhausted") from inner
    except RuntimeError as outer:
        assert main._is_capacity_error(outer) is True


def test_detects_status_code_429() -> None:
    class Boom(Exception):
        status_code = 429

    assert main._is_capacity_error(Boom("no message hint")) is True


def test_ignores_ordinary_error() -> None:
    assert main._is_capacity_error(ValueError("bad input")) is False
    assert main._is_capacity_error(KeyError("missing")) is False

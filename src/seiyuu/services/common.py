"""Shared service error type — one catchable class for CLI/API boundary mapping."""


class ServiceError(Exception):
    """Loud service failure with an actionable, user-facing message."""

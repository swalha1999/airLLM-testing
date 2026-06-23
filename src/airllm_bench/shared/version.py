"""Project version tracking (V3 S8).

Versions start at 1.00 and increment on meaningful changes. This is the single
source of truth for the harness code version; config files carry their own
``"version"`` keys validated separately at startup.
"""

__version__ = "1.00"


def get_version() -> str:
    """Return the current harness code version string."""
    return __version__

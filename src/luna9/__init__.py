"""Luna9 package root."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("luna9")
except PackageNotFoundError:
    __version__ = "0.1.0"

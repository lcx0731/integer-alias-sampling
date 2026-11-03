"""Integer-weight sampling with replacement. No runtime dependencies."""

from .core import AliasTable, SamplingError, build, uniform_below

__all__ = ["AliasTable", "SamplingError", "build", "uniform_below"]

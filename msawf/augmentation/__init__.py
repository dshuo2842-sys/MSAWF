"""Paper-traceable prefix and insertion-like transformations."""

from .insertion import InsertionResult, InsertionTransform
from .prefix import PrefixOperator, RandomPrefixSampler

__all__ = [
    "InsertionResult",
    "InsertionTransform",
    "PrefixOperator",
    "RandomPrefixSampler",
]

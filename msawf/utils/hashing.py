"""Stable SHA-256 primitives for protocol identity and deterministic RNG."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON-compatible data with a path-independent canonical layout."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_hex(value: Any) -> str:
    """Return the SHA-256 digest of canonical JSON-compatible data."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _feed_part(hasher: Any, value: Any) -> None:
    encoded = canonical_json_bytes(value)
    hasher.update(len(encoded).to_bytes(8, "big"))
    hasher.update(encoded)


def stable_digest(domain: str, *parts: Any) -> str:
    """Hash domain-separated, length-prefixed values without built-in ``hash``."""

    if not domain:
        raise ValueError("domain must not be empty")
    hasher = hashlib.sha256()
    _feed_part(hasher, "msawf-stable-digest-v1")
    _feed_part(hasher, domain)
    for part in parts:
        _feed_part(hasher, part)
    return hasher.hexdigest()


def derive_seed(domain: str, *parts: Any) -> int:
    """Derive an unsigned 64-bit seed using domain-separated SHA-256."""

    return int.from_bytes(bytes.fromhex(stable_digest(domain, *parts))[:8], "big")


def derive_five_split_seed(
    base_seed: int, dataset_fingerprint: str, split_id: str
) -> int:
    """Implement the approved ``msawf-five-split-v1`` seed formula."""

    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    if not dataset_fingerprint:
        raise ValueError("dataset_fingerprint must not be empty")
    if not split_id:
        raise ValueError("split_id must not be empty")
    hasher = hashlib.sha256()
    hasher.update(b"msawf-five-split-v1")
    hasher.update(base_seed.to_bytes(8, "big", signed=False))
    hasher.update(dataset_fingerprint.encode("utf-8"))
    hasher.update(split_id.encode("utf-8"))
    return int.from_bytes(hasher.digest()[:8], "big")


class StableRNG:
    """Counter-based SHA-256 RNG with deterministic integer sampling."""

    _LIMIT = 1 << 64

    def __init__(self, seed: int) -> None:
        if seed < 0 or seed >= self._LIMIT:
            raise ValueError("seed must be an unsigned 64-bit integer")
        self._seed = seed
        self._counter = 0

    def _next_uint64(self) -> int:
        hasher = hashlib.sha256()
        hasher.update(b"msawf-stable-rng-v1")
        hasher.update(self._seed.to_bytes(8, "big"))
        hasher.update(self._counter.to_bytes(16, "big"))
        self._counter += 1
        return int.from_bytes(hasher.digest()[:8], "big")

    def randbelow(self, upper_bound: int) -> int:
        """Return a uniform integer in ``[0, upper_bound)`` by rejection sampling."""

        if upper_bound <= 0:
            raise ValueError("upper_bound must be positive")
        if upper_bound > self._LIMIT:
            raise ValueError("upper_bound must not exceed 2**64")
        acceptance_limit = self._LIMIT - (self._LIMIT % upper_bound)
        while True:
            value = self._next_uint64()
            if value < acceptance_limit:
                return value % upper_bound

    def weighted_index(self, weights: tuple[int, ...] | list[int]) -> int:
        """Choose an index with probability proportional to integer weights."""

        if not weights or any(weight < 0 for weight in weights):
            raise ValueError("weights must be a non-empty sequence of non-negative values")
        total = sum(weights)
        if total <= 0:
            raise ValueError("at least one weight must be positive")
        draw = self.randbelow(total)
        cumulative = 0
        for index, weight in enumerate(weights):
            cumulative += weight
            if draw < cumulative:
                return index
        raise RuntimeError("weighted selection failed")

"""Approved deterministic shuffle, batching, and Stage II cycling."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Generic, Iterator, Sequence, TypeVar

import torch

from msawf.utils.hashing import StableRNG, derive_seed

ItemT = TypeVar("ItemT")
BatchT = TypeVar("BatchT")
SourceBatchT = TypeVar("SourceBatchT")
TargetBatchT = TypeVar("TargetBatchT")


def _shuffle_indices(size: int, seed: int) -> list[int]:
    indices = list(range(size))
    rng = StableRNG(seed)
    for position in range(size - 1, 0, -1):
        swap = rng.randbelow(position + 1)
        indices[position], indices[swap] = indices[swap], indices[position]
    return indices


class DeterministicBatchSampler:
    """Shuffle once per epoch/cycle and preserve every record without singletons."""

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        *,
        base_seed: int,
        domain: str,
        epoch: int,
        cycle: int = 0,
    ) -> None:
        if dataset_size < 2:
            raise ValueError("training dataset must contain at least two eligible records")
        if batch_size < 2:
            raise ValueError("training batch_size must be at least two")
        if base_seed < 0:
            raise ValueError("base_seed must be non-negative")
        if not domain.strip():
            raise ValueError("shuffle domain must not be empty")
        if epoch < 1:
            raise ValueError("epoch must be one-based and positive")
        if cycle < 0:
            raise ValueError("cycle must be non-negative")
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.base_seed = base_seed
        self.domain = domain
        self.epoch = epoch
        self.cycle = cycle
        self.seed = derive_seed(
            "trainer-loader-shuffle-v1", base_seed, domain, epoch, cycle
        )

    def _batches(self) -> list[list[int]]:
        indices = _shuffle_indices(self.dataset_size, self.seed)
        batches = [
            indices[start : start + self.batch_size]
            for start in range(0, self.dataset_size, self.batch_size)
        ]
        if len(batches[-1]) == 1:
            if len(batches) == 1:
                raise RuntimeError("singleton dataset should have failed validation")
            if len(batches[-2]) > 2:
                batches[-1].insert(0, batches[-2].pop())
            else:
                batches[-2].extend(batches.pop())
        if any(len(batch) < 2 for batch in batches):
            raise RuntimeError("batch rebalance produced a singleton")
        return batches

    @property
    def batch_sizes(self) -> tuple[int, ...]:
        return tuple(len(batch) for batch in self._batches())

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches()

    def __len__(self) -> int:
        quotient, remainder = divmod(self.dataset_size, self.batch_size)
        if remainder == 0:
            return quotient
        if remainder == 1 and quotient > 0 and self.batch_size == 2:
            return quotient
        return quotient + 1


class DeterministicBatchLoader(Generic[ItemT, BatchT]):
    """Small loader abstraction whose epoch and cycle are explicit inputs."""

    def __init__(
        self,
        dataset: Sequence[ItemT],
        *,
        batch_size: int,
        base_seed: int,
        domain: str,
        collate_fn: Callable[[list[ItemT]], BatchT],
    ) -> None:
        if len(dataset) < 2:
            raise ValueError("training dataset must contain at least two eligible records")
        self.dataset = dataset
        self.batch_size = batch_size
        self.base_seed = base_seed
        self.domain = domain
        self.collate_fn = collate_fn

    def sampler(self, *, epoch: int, cycle: int = 0) -> DeterministicBatchSampler:
        return DeterministicBatchSampler(
            len(self.dataset),
            self.batch_size,
            base_seed=self.base_seed,
            domain=self.domain,
            epoch=epoch,
            cycle=cycle,
        )

    def iter_epoch(self, *, epoch: int, cycle: int = 0) -> Iterator[BatchT]:
        for indices in self.sampler(epoch=epoch, cycle=cycle):
            yield self.collate_fn([self.dataset[index] for index in indices])

    def __len__(self) -> int:
        return len(self.sampler(epoch=1))


@dataclass(frozen=True)
class PairedBatch(Generic[SourceBatchT, TargetBatchT]):
    source: SourceBatchT
    target: TargetBatchT
    step: int
    source_cycle: int
    target_cycle: int


def iter_cycled_pairs(
    source_loader: DeterministicBatchLoader[object, SourceBatchT],
    target_loader: DeterministicBatchLoader[object, TargetBatchT],
    *,
    epoch: int,
) -> Iterator[PairedBatch[SourceBatchT, TargetBatchT]]:
    """Traverse the longer loader once and deterministically cycle the shorter."""

    steps = max(len(source_loader), len(target_loader))
    source_cycle = 0
    target_cycle = 0
    source_iterator = source_loader.iter_epoch(epoch=epoch, cycle=source_cycle)
    target_iterator = target_loader.iter_epoch(epoch=epoch, cycle=target_cycle)
    for step in range(steps):
        try:
            source = next(source_iterator)
        except StopIteration:
            source_cycle += 1
            source_iterator = source_loader.iter_epoch(epoch=epoch, cycle=source_cycle)
            source = next(source_iterator)
        try:
            target = next(target_iterator)
        except StopIteration:
            target_cycle += 1
            target_iterator = target_loader.iter_epoch(epoch=epoch, cycle=target_cycle)
            target = next(target_iterator)
        yield PairedBatch(
            source=source,
            target=target,
            step=step,
            source_cycle=source_cycle,
            target_cycle=target_cycle,
        )


@dataclass(frozen=True)
class WorkerSeedInitializer:
    """Picklable Windows-safe DataLoader worker initializer."""

    base_seed: int
    stage: str
    domain: str
    split_id: str
    epoch: int
    cycle: int = 0

    def __call__(self, worker_id: int) -> None:
        seed = derive_seed(
            "trainer-worker-v1",
            self.base_seed,
            self.stage,
            self.domain,
            self.split_id,
            self.epoch,
            self.cycle,
            worker_id,
        )
        random.seed(seed)
        torch.manual_seed(seed)
        try:
            import numpy as np

            np.random.seed(seed % (2**32))
        except (ImportError, OSError):
            pass

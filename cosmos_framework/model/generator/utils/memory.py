# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Abstract interfaces for persistent memory in the MoT transformer stack.

``MemoryState`` is a *mutable* Python object that lives **outside** the
``torch.compile`` boundary.  It is responsible for reading cached tensors
(``read_for_layer``) and writing new tensors back (``write_for_layer``).

``MemoryValue`` is a *read-only* tensor container that is safe to pass
**into** a compiled decoder layer.  Concrete implementations are plain
dataclasses whose fields are tensors (or None).  No methods on
``MemoryValue`` should mutate state.

``KVToStore`` is a type alias for the 4-tuple of tensors
``(gen_k, gen_v, und_k, und_v)`` returned by each compiled layer so
the caller can write them back into the cache outside the compile boundary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

# (gen_k, gen_v, und_k, und_v) returned by each compiled layer for the caller
# to write back into the cache outside the torch.compile boundary.
KVToStore = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass
class MemoryValue(ABC):
    """Read-only tensor container safe to pass into ``torch.compile``.

    Concrete subclasses (e.g. ``ARMemoryValue``, ``KVTrainMemoryValue``)
    are plain dataclasses of tensors.  No methods on this class should
    mutate state or perform non-trivial computation.
    """

    @property
    def supports_context_parallel_attention(self) -> bool:
        """Whether this memory value is compatible with context-parallel attention.

        Overridden by ``KVTrainMemoryValue`` to return ``False``.  Used by
        ``ContextParallelDispatch`` to reject an unsupported combination
        without importing the concrete subclass.
        """
        return True


class MemoryState(ABC):
    """Mutable persistent memory that lives outside ``torch.compile``.

    The outer loop in ``_impl_forward`` calls ``read_for_layer`` before
    each decoder layer and ``write_for_layer`` after.  The ``MemoryState``
    object itself is **never** passed into a compiled region.
    """

    @abstractmethod
    def init(self, hidden_states: dict, device: torch.device) -> None:
        """Initialization per training step.

        Called once before any transformer layers are processed.

        Args:
            hidden_states: The packed sequence (``SequencePack``).
            device: Target device for any new tensors.
        """

    @abstractmethod
    def read_for_layer(self, layer_idx: int) -> MemoryValue:
        """Produce a read-only tensor snapshot for *layer_idx*.

        Used to retrieve KV values from the cache.
        The returned ``MemoryValue`` is passed into the compiled decoder
        layer as ``memory_value``.
        """

    @abstractmethod
    def write_for_layer(self, layer_idx: int, kv_to_store: KVToStore) -> None:
        """Store the K/V tensors produced by *layer_idx* back into the cache.

        Called outside the ``torch.compile`` boundary.
        """

    @abstractmethod
    def is_gen_only(self) -> bool:
        """Return ``True`` when only the generation pathway should run.

        When ``True``, the decoder layer assumes that the text caption has
        already been processed and cached in the MemoryState object.
        Used for autoregressive frame-by-frame generation of video.
        """

    def requires_natten_metadata(self) -> bool:
        """Whether the packed-sequence builder should create NATTEN metadata.

        Memory paths whose attention implementation handles temporal
        visibility itself return ``False``.
        """
        return True


@dataclass
class ConditionKVCacheValue(MemoryValue):
    """Per-layer understanding K/V reused across denoising steps."""

    und_k: torch.Tensor | None = None
    und_v: torch.Tensor | None = None
    frame_idx: int = 0


class ConditionKVCacheState(MemoryState):
    """Request-local cache for an invariant understanding condition.

    The first network forward populates one understanding K/V pair per layer.
    Later denoising steps execute only the generation tower. A separate state
    must be used for each CFG branch because their text conditions differ.
    """

    def __init__(self) -> None:
        self._layers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._gen_only = False

    def init(self, hidden_states: dict, device: torch.device) -> None:
        del hidden_states, device
        self._gen_only = bool(self._layers)

    def read_for_layer(self, layer_idx: int) -> ConditionKVCacheValue:
        cached = self._layers.get(layer_idx)
        if cached is None:
            if self._gen_only:
                raise RuntimeError(f"Condition K/V cache is incomplete at layer {layer_idx}")
            return ConditionKVCacheValue()
        und_k, und_v = cached
        return ConditionKVCacheValue(und_k=und_k, und_v=und_v, frame_idx=1)

    def write_for_layer(self, layer_idx: int, kv_to_store: KVToStore) -> None:
        if self._gen_only:
            return
        _, _, und_k, und_v = kv_to_store
        self._layers[layer_idx] = (und_k.detach(), und_v.detach())

    def is_gen_only(self) -> bool:
        return self._gen_only

    def requires_natten_metadata(self) -> bool:
        return False

    def clear(self) -> None:
        self._layers.clear()
        self._gen_only = False

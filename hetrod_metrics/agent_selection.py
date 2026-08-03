"""Shared eligibility helpers for HetroD interaction selection."""

from __future__ import annotations

import torch

from .config import DEFAULT_CONFIG, HetrodMetricConfig


def future_valid_frames(
    track_masks: torch.Tensor,
    config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> torch.Tensor:
    """Count valid future frames for every agent."""
    return track_masks[:, config.future_start_index :].sum(dim=1)


def is_evaluated_type(
    object_types: torch.Tensor,
    config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> torch.Tensor:
    """Return the vehicle, two-wheeler, and pedestrian eligibility mask."""
    valid_types = torch.tensor(
        config.evaluated_object_types,
        device=object_types.device,
        dtype=object_types.dtype,
    )
    return torch.isin(object_types, valid_types)

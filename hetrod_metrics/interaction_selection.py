"""Official v0.8 interaction selector for the HetroD Challenge.

v0.8 configures the shared candidate evidence and hard caps with strict
semantic buckets used for diverse ranking:

* static means locally static near the conflict, not short over the full track;
* following/overtake require persistent same-corridor geometry;
* merges are identified before order swaps can be called overtakes;
* overtakes use their dedicated Tier-B gate;
* low-motion pairs are context, not interaction targets;
* fill ranking prefers unseen behavior and pair types after strength tier.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .config import DEFAULT_CONFIG, HetrodMetricConfig
from .interaction_candidates import AgentSelectionResult
from .interaction_ranking import (
    DiverseSelectionConfig,
    select_diverse_interaction_agents,
)


OFFICIAL_SELECTION_CONFIG = replace(
    DiverseSelectionConfig(),
    use_local_static_semantics=True,
    require_corridor_persistence=True,
    overtake_uses_dedicated_tier=True,
    reject_low_motion_pairs=True,
    prefer_pair_type_diversity=True,
    allow_static_cap_for_new_anchor_type=True,
)


def select_interaction_agents(
    gt_scenario: dict[str, Any],
    diverse_config: DiverseSelectionConfig = OFFICIAL_SELECTION_CONFIG,
    metric_config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> AgentSelectionResult:
    """Select strong, semantically constrained, behavior-diverse pairs."""
    return select_diverse_interaction_agents(
        gt_scenario,
        diverse_config=diverse_config,
        metric_config=metric_config,
    )


def select_competition_agents(
    gt_scenario: dict[str, Any],
    metric_config: HetrodMetricConfig = DEFAULT_CONFIG,
) -> AgentSelectionResult:
    """Competition-facing wrapper with the evaluator's stable call shape."""
    selection_config = replace(
        OFFICIAL_SELECTION_CONFIG,
        max_pairs=metric_config.selection_max_pairs,
        max_pair_endpoints=metric_config.selection_max_pair_endpoints,
        max_pairs_per_agent=metric_config.selection_max_pairs_per_agent,
        min_anchor_path_length_m=(
            metric_config.selection_min_anchor_path_length_m
        ),
        min_anchor_motion_extent_m=(
            metric_config.selection_min_anchor_motion_extent_m
        ),
        min_history_valid_frames=(
            metric_config.selection_min_history_valid_frames
        ),
        num_fallback_agents=metric_config.selection_num_fallback_agents,
        min_fallback_agents=metric_config.selection_min_fallback_agents,
    )
    return select_interaction_agents(
        gt_scenario,
        diverse_config=selection_config,
        metric_config=metric_config,
    )

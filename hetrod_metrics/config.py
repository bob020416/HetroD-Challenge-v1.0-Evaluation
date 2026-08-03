from __future__ import annotations

from dataclasses import dataclass

from waymo_open_dataset.protos import scenario_pb2


@dataclass(frozen=True)
class HetrodMetricConfig:
    current_time_index: int = 10
    future_start_index: int = 11
    required_num_rollouts: int = 32
    min_future_valid_frames: int = 20
    selection_min_history_valid_frames: int = 5
    selection_min_anchor_path_length_m: float = 2.0
    selection_min_anchor_motion_extent_m: float = 1.0
    selection_path_margin_m: float = 0.5
    selection_max_arrival_gap_s: float = 3.0
    selection_max_synchronous_distance_m: float = 5.0
    selection_min_motion_response_score: float = 0.4
    selection_min_direct_closing_distance_m: float = 2.0
    selection_min_direct_motion_response_score: float = 0.2
    selection_max_pairs: int = 4
    selection_max_pair_endpoints: int = 8
    selection_max_pairs_per_agent: int = 2
    selection_num_fallback_agents: int = 4
    selection_min_fallback_agents: int = 2
    cross_type_pair_distance_gate_m: float = 10.0
    cross_type_pair_chunk_size: int = 2048
    collision_pair_chunk_size: int = 256
    seconds_per_step: float = 0.1
    vehicle_valid_region_margin_m: float = 0.0
    two_wheeler_valid_region_margin_m: float = 1.0
    pedestrian_valid_region_margin_m: float = 2.0
    pedestrian_gt_road_corridor_margin_m: float = 1.5
    pedestrian_transition_time_slack_s: float = 1.0
    valid_region_agent_chunk_size: int = 4
    valid_region_query_chunk_size: int = 4096
    coverage_grid_resolution_m: float = 0.5
    coverage_map_query_chunk_size: int = 1024
    coverage_default_vehicle_length_m: float = 4.5
    coverage_default_vehicle_width_m: float = 1.8
    coverage_default_two_wheeler_length_m: float = 1.8
    coverage_default_two_wheeler_width_m: float = 0.7
    coverage_default_pedestrian_length_m: float = 0.8
    coverage_default_pedestrian_width_m: float = 0.8

    kinematic_weight: float = 0.30
    safety_weight: float = 0.35
    cross_type_weight: float = 0.25
    coverage_weight: float = 0.10

    vehicle_type: int = scenario_pb2.Track.ObjectType.TYPE_VEHICLE
    pedestrian_type: int = scenario_pb2.Track.ObjectType.TYPE_PEDESTRIAN
    two_wheeler_type: int = scenario_pb2.Track.ObjectType.TYPE_CYCLIST

    @property
    def evaluated_object_types(self) -> tuple[int, int, int]:
        return (self.vehicle_type, self.two_wheeler_type, self.pedestrian_type)


DEFAULT_CONFIG = HetrodMetricConfig()

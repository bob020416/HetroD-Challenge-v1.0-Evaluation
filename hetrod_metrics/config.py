from __future__ import annotations

from dataclasses import dataclass

from waymo_open_dataset.protos import scenario_pb2


@dataclass(frozen=True)
class HetrodMetricConfig:
    current_time_index: int = 10
    future_start_index: int = 11
    future_displacement_threshold_m: float = 1.0
    valid_region_distance_threshold_m: float = 1.0
    cross_type_distance_threshold_m: float = 5.0
    cross_type_ttc_threshold_s: float = 4.0
    cross_type_ttc_proximity_radius_m: float = 5.0
    cross_type_pair_distance_gate_m: float = 10.0
    cross_type_pair_ttc_gate_s: float = 6.0
    cross_type_distance_hist_max_m: float = 15.0
    cross_type_ttc_hist_max_s: float = 6.0
    cross_type_distance_hist_bins: int = 30
    cross_type_ttc_hist_bins: int = 24
    seconds_per_step: float = 0.1
    collision_tolerance_m: float = 0.1
    vehicle_valid_region_margin_m: float = 0.0
    two_wheeler_valid_region_margin_m: float = 1.0
    pedestrian_valid_region_margin_m: float = 2.0
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

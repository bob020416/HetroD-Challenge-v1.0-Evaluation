# HetroD Metrics

## Agent Selection

Evaluate agent `i` if:

```text
has_full_history(i)
has_full_future(i)
type_i in {vehicle, two_wheeler, pedestrian}
future_displacement(i) > 1.0 m
distance_to_valid_region(i) < 1.0 m
min_cross_type_distance(i) < 5.0 m OR min_cross_type_TTP(i) < 4.0 s
```

`TTP` is constant-velocity time to enter a 5 m proximity radius.

## Score

```text
Base =
0.30 Kinematic
+ 0.35 Safety
+ 0.25 Cross-type

Coverage Bonus =
0.10 * Coverage * Kinematic * Safety

Overall = Base + Coverage Bonus
```

## Kinematic

```text
Kinematic = mean(
  Linear Speed,
  Linear Acceleration,
  Angular Speed,
  Angular Acceleration
)
```

Each metric uses the WOSAC likelihood estimator, normalizes by the GT-as-rollout
ceiling, then macro-averages vehicle / two-wheeler / pedestrian.

## Safety

```text
Safety =
0.5 Collision with Annotation Tolerance
+ 0.5 Valid Region Margin
```

- Collision ignores overlap shallower than `0.1 m`.
- Valid region uses road-edge signed distance fallback:
  vehicle `0 m`, two-wheeler `1 m`, pedestrian `2 m`.

## Cross-Type

```text
Cross-type =
0.5 Distance Proximity to GT
+ 0.5 Time-to-Proximity Proximity to GT
```

Only unique physical cross-type pairs are used. Pair types are
vehicle-pedestrian, vehicle-two-wheeler, and pedestrian-two-wheeler. Candidate
pairs must have GT distance `< 10 m` or GT TTP `< 6 s`.

## Coverage

Rasterize selected-agent oriented boxes on a `0.5 m` BEV grid.

Per agent/timestep:

```text
incremental coverage =
(union cells - largest single-rollout footprint)
/
(sum rollout footprint cells - largest single-rollout footprint)
```

Identical rollouts score `0`. Fully non-overlapping valid footprints score `1`.
Scores are averaged by agent, type, then present types. Missing pedestrian size
uses `0.8 m x 0.8 m`.

## Aggregation

Dataset aggregation is type-balanced:

- Kinematic/Safety: aggregate samples within each agent type, then macro-average.
- Cross-type: aggregate within each pair type, then macro-average.
- Coverage: aggregate within each agent type, then macro-average.

No-selected scenarios are reported as `skipped_no_selected_agents` and excluded
from score aggregation.

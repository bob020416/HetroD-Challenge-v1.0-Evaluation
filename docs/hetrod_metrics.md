# HetroD Metrics (`hetrod-0.2.0`)

## Agent Selection

Evaluate agent `i` if:

```text
type_i in {vehicle, two_wheeler, pedestrian}
AND current_frame_valid(i)
AND full_history_valid(i)
AND future_valid_frames(i) >= 20
AND (
  min_cross_type_distance(i) < 5.0 m
  OR min_cross_type_TTP(i) < 4.0 s
)
```

The interaction gate is computed from GT and only decides which agents are
relevant. TTP is constant-velocity time to enter a 5 m proximity radius.
Partial future tracks remain eligible; every metric ignores GT-invalid frames.
Selection does not depend on motion amount or map position.

## Score

```text
Base =
0.30 Kinematic
+ 0.35 Safety
+ 0.25 Cross-type

Diversity Bonus =
0.10 * Diversity * Kinematic * Safety

Overall = Base + Diversity Bonus
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

Each submission provides exactly 32 rollouts. Kinematic metrics use the WOSAC
likelihood estimator, normalize by the GT-as-rollout ceiling, then
macro-average vehicle / two-wheeler / pedestrian.

## Safety

```text
Safety = 0.5 Collision + 0.5 Valid Region
```

Collision has no tolerance. For each selected agent and rollout, it is `1` if
the oriented box strictly overlaps any other valid agent at least once on a
frame where both GT tracks are valid. Therefore:

```text
Collision score =
1 - collided_agent_rollouts / valid_agent_rollouts
```

Valid region uses road-edge signed distance with type margins: vehicle `0 m`,
two-wheeler `1 m`, pedestrian `2 m`. It penalizes only excess outside-region
frequency relative to the annotation:

```text
Valid-region score =
1 - max(sim_outside_rate - GT_outside_rate, 0)
```

This prevents known map/annotation disagreement from lowering a perfect GT
replay's score.

## Cross-Type Interaction

For each unique physical cross-type pair and rollout, find closest center
distance `d_min` and its time `t*`. Include a pair if GT or any simulated
rollout comes within 10 m; the union catches both missed and invented
interactions.

```text
distance score = max(0, 1 - |sim_d_min - GT_d_min| / 5 m)
time score     = max(0, 1 - |sim_t* - GT_t*| / 4 s)
pair score     = 0.5 * distance score + 0.5 * time score
```

Scores are averaged over rollouts and pairs, then macro-averaged across
vehicle-pedestrian, vehicle-two-wheeler, and pedestrian-two-wheeler. This
metric does not use histogram bins or a second constant-velocity TTP model.

## Diversity Bonus

Rasterize selected-agent oriented boxes on a `0.5 m` BEV grid. Identical
rollouts score `0`; separated valid footprints increase the score. Diversity
remains a bonus gated by Kinematic and Safety, so unrealistic or unsafe spread
cannot compensate for poor base quality.

## Aggregation

Within each location:

- Kinematic/Safety aggregate within agent type, then macro-average types.
- Cross-type aggregates within pair type, then macro-averages pair types.
- Diversity aggregates within agent type, then macro-averages types.

The leaderboard score is then an equal macro-average of all locations present
in the split. No-selected scenarios are reported as
`skipped_no_selected_agents` and excluded.

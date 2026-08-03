# HetroD Metrics (`hetrod-0.8.0`)

## Agent Selection

Selection first finds GT interaction pairs, then separates moving scoring
**anchors** from non-scored **context**. An agent can be an anchor if:

```text
type_i in {vehicle, two_wheeler, pedestrian}
AND current_frame_valid(i)
AND history_valid_frames(i) >= 5
AND future_valid_frames(i) >= 20
AND valid_future_path_length(i) >= 2.0 m
AND belongs_to_a_qualified_interaction_pair(i)
```

A qualified pair uses full oriented footprints and requires temporal
relevance plus interpretable evidence from swept-path arrival time, direct
proximity, closing motion, or braking/turning response. Swept-path conflicts
use an arrival gap below `3.0 s`; the selection-only footprint margin is
`0.5 m`. Same-type pairs are allowed for selecting anchors.

The `hetrod-0.8.0` selector ranks only Tier A/B candidates and keeps at most
four pairs, eight endpoints, and degree two per endpoint. It first preserves
the strongest complex and cross-type evidence, then prefers unseen behavior
and pair types. Following and overtake require persistent same-corridor
geometry; an overtake additionally requires an order swap and uses a dedicated
Tier-B gate. Static means local speed at the interaction is at most `0.30 m/s`,
not merely a short full trajectory. Low-motion pairs are not interaction
targets. A slow/static member remains collision context when it is not itself
a moving scoring anchor.

Following and parallel are capped at one pair per scenario. Moving-static is
also capped at one, except that a second pair may be retained when it adds a
previously uncovered scoring agent type. Complex behavior families are capped
at two pairs each.

If no qualified pair exists, two to four eligible agents are selected as
`fallback_noninteractive`, preferring type diversity and then future motion.
This keeps kinematic/safety evaluation defined without inventing an
interaction. Cross-type interaction is N/A for such a scenario and its quality
weight is redistributed proportionally across kinematic and safety. Selection
uses reference GT only and is identical for every submission.

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

Each submission provides exactly 32 rollouts. Kinematic metrics use the WOSAC
likelihood estimator, normalize by the GT-as-rollout ceiling, then
macro-average vehicle / two-wheeler / pedestrian.

## Safety

```text
Safety = 0.5 Collision + 0.5 Valid Region
```

Collision has no tolerance. A collision is new when a simulated pair strictly
overlaps on a frame where the same physical GT pair does not overlap. For each
selected agent and rollout, any new pair/frame overlap marks that rollout as
collided. Therefore:

```text
Collision score =
1 - newly_collided_agent_rollouts / valid_agent_rollouts
```

This pair/frame GT conditioning prevents annotation overlaps already present in
GT from lowering a perfect replay's score. It does not add geometric tolerance
or exempt collisions with a different partner or at a different frame.

Valid region uses schema 1.3 semantic layers embedded in HetroD GT:

- vehicle: road/intersection/parking/emergency-lane surfaces with a `0.75 m`
  boundary margin;
- two-wheeler: vehicle surfaces plus bicycle-specific surfaces with a `0.75 m`
  boundary margin;
- pedestrian permanent core: walkway/footway/pedestrian/freespace surfaces
  with a `0.75 m` boundary margin;
- pedestrian transition: expanded crosswalk surfaces and that pedestrian's
  own GT-supported road corridor with a `1.5 m` margin.

Vehicle and two-wheeler footprints use the static type-specific rule.
Pedestrian core is always valid. Crosswalk/corridor occupancy is valid up to
the pedestrian's GT transition duration plus `1.0 s`; additional transition
frames are penalized. A simulated pedestrian is spatially offroad when its
full footprint is in neither core, crosswalk, nor its own GT corridor.

```text
pedestrian penalty =
  spatial_offroad_frames
  + max(0, simulated_transition_frames - GT_transition_frames - 10)

Pedestrian valid-region score =
1 - pedestrian_penalty / map_supported_GT_valid_frames
```

GT frames outside every mapped semantic layer are `map_unsupported`: they are
excluded locally from both numerator and denominator and reported as
`excluded_map_unsupported_rate`. They are not automatically turned into valid
corridors. Missing annotated dimensions use stable type defaults. GT files
without schema 1.3 remain supported through the schema 1.2 polygon or legacy
road-edge fallback.

## Cross-Type Interaction

For each reference-selected physical pair whose two types differ, find closest
center distance `d_min` and its time `t*` in every rollout. Same-type selection
pairs still inform anchor/context and collision evaluation but never enter this
component. Unselected nearby agents are not silently added to the cross-type
metric.

```text
distance score = max(0, 1 - |sim_d_min - GT_d_min| / 5 m)
time score     = max(0, 1 - |sim_t* - GT_t*| / 4 s)
pair score     = 0.5 * distance score + 0.5 * time score
```

All explicitly selected cross-type pairs are included. Scores are averaged
over rollouts and pairs, then macro-averaged across
vehicle-pedestrian, vehicle-two-wheeler, and pedestrian-two-wheeler. This
metric does not use histogram bins or a second constant-velocity TTP model.

## Coverage Bonus

Rasterize selected-agent oriented boxes on a `0.5 m` BEV grid and retain only
cells inside the corresponding type-specific polygon. Identical rollouts score
`0`; separated valid footprints increase the coverage score. Coverage remains
a bonus gated by Kinematic and Safety, so unrealistic or unsafe spread cannot
compensate for poor base quality.

## Aggregation

Within each location:

- Kinematic/Safety aggregate within agent type, then macro-average types.
- Cross-type aggregates within pair type, then macro-averages pair types.
- Diversity aggregates within agent type, then macro-averages types.

The leaderboard score is then an equal macro-average of all locations present
in the split. No-selected scenarios are reported as
`skipped_no_selected_agents` and excluded.

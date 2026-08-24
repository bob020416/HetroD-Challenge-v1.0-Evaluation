# Dataset

## Download

Download the public package from the
[HetroD dataset page](https://levelxdata.com/hetrod-dataset/).

## Layout

```text
HetroD-Challenge-v1.0-public/
  manifests/
    train.txt
    valid.txt
    test.txt
    train_gt_paths.txt
    valid_gt_paths.txt
    train_scenarionet_paths.txt
    valid_scenarionet_paths.txt
    test_input_paths.txt
  train/
    gt/
    scenarionet/
  valid/
    gt/
    scenarionet/
  test/
    input/
  split_summary.json
  public_release_summary.json
```

Counts:

```text
train: 5087
valid:  955
test:   955
```

Test GT is hidden. Public test files are ScenarioNet-compatible inputs with
future targets masked.

## IDs

Scenario ID:

```text
<date>_loc<location>_seg<segment>_ego_<ego_id>
```

Files:

```text
GT:          <scenario_id>.pkl
ScenarioNet: sd_HetroD_1.0_<scenario_id>.pkl
```

All manifest paths are relative to the package root.

## GT Pickle Schema

Train/valid GT pickles contain:

```text
scenario_id
timestamps_seconds
current_time_index
sdc_track_index
objects_of_interest
tracks
track_masks
object_ids
object_types
road_edges
predict_index
sim_agent_ids
predict_agent_ids
lane_ids
lane_polylines
traffic_signals
valid_regions
valid_region_definition
```

Shapes:

```text
tracks:       [num_agents, 91, 9]
track_masks:  [num_agents, 91]
object_ids:   [num_agents]
object_types: [num_agents]
```

`tracks[..., :]`:

```text
x, y, z, length, width, height, yaw, velocity_x, velocity_y
```

`object_ids`, `sim_agent_ids`, and `predict_agent_ids` are the required
submitted agent set.

`valid_regions` stores schema 1.3 polygons for `vehicle`, `cyclist`,
`pedestrian_core`, `pedestrian_crosswalk`, and `pedestrian_road`; a legacy
`pedestrian` union remains for compatible map consumers. Each polygon record
contains an `exterior` tensor and a list of `holes`. Standard margins are
vehicle/cyclist/core `0.75 m`, crosswalk `1.5 m`, and raw road `0.0 m`.
Evaluation constructs a `1.5 m` agent-specific road corridor from hidden GT
and permits the GT transition duration plus `1.0 s`. GT frames outside every
semantic layer are excluded locally and reported as map unsupported.

## Test Input

Use:

```text
scenario_id: top-level id
agent_id:    metadata.required_agent_ids
```

Submit one `.pkl` rollout for every scenario in `manifests/test.txt`, as in the
original challenge specification. Each pickle contains `agent_id` and
`simulated_states`; states use shape `[32, num_agents, 80, 4]` in global
`(x_m, y_m, z_m, yaw_rad)`. Numeric `.npz` files with the same keys are
accepted as an optional alternative.

Archive:

```text
your_team_submission.zip
  your_team_submission/
    <scenario_id_0>.pkl
    <scenario_id_1>.pkl
    ...
```

The submission portal, schedule, and current challenge rules are maintained on
the [HetroD dataset page](https://levelxdata.com/hetrod-dataset/).

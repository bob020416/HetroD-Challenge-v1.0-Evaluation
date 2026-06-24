# Dataset

## Download

```text
Public data link: TBD
```

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

## Test Input

Use:

```text
scenario_id: top-level id
agent_id:    metadata.required_agent_ids
```

Submit one rollout pickle for every scenario in `manifests/test.txt`.
Each pickle must use `simulated_states` shape `[32, num_agents, 80, 4]`.

Archive:

```text
your_team_submission.zip
  your_team_submission/
    <scenario_id_0>.pkl
    <scenario_id_1>.pkl
    ...
```

```text
Submission Google Form: TBD
Deadline: September 5, 2026 (AoE)
```

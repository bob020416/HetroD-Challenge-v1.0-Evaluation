# HetroD Challenge Evaluation Toolkit

Offline HetroD Challenge evaluator built on the fast WOSAC metric backend.

## Install

```bash
conda create -y -n wosac_eval python=3.11.9
conda activate wosac_eval
pip install -r requirements.txt
pip install --no-deps waymo-open-dataset-tf-2-12-0==1.6.7
```


## Data

```text
Public data link: https://drive.google.com/file/d/1FvyIpt0I-GVdYoKR24Yus_NJz9Gi4J3v/view?usp=sharing

Please request access using your official institutional or company Google Account. In the request message, include:

- Full name
- Institution or company
- Position
- Team name
- Official institutional email
- Type: I confirm that I have read and agree to the HetroD Challenge Data Licence v1.0.
- Type: I confirm that I will not redistribute or commercially use the dataset.

Incomplete or unverifiable requests may be declined.
```

Public package layout:

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
  train/{gt,scenarionet}/
  valid/{gt,scenarionet}/
  test/input/
  split_summary.json
  public_release_summary.json
```

Split:

```text
train: 5087
valid:  955
test:   955
```

Test GT is hidden. Test input keeps history/map data and masks future targets.
See [docs/dataset.md](docs/dataset.md).

## Validate Rollouts

Run inference on `valid/scenarionet`, write rollout pickles, then evaluate:

```bash
python hetrod_eval.py /path/to/valid_rollouts \
  --gt-dir /path/to/HetroD-Challenge-v1.0-public/valid/gt \
  --output valid_hetrod_metrics_report.json \
  --device cuda
```

## Submission

Rollout pickle name:

```text
<scenario_id>.pkl
```

Rollout pickle content:

```python
{
    "agent_id": ...,          # [num_agents]
    "simulated_states": ...,  # [32, num_agents, 80, 4]
}
```

`simulated_states` contains exactly 32 rollouts in global `(x, y, z, yaw)` for
future timesteps 11..90.

For public test input:

- `scenario_id`: top-level input `id`, also listed in `manifests/test.txt`
- required agents: `metadata.required_agent_ids`
- output one pickle for every test scenario

Final archive:

```text
your_team_submission.zip
  your_team_submission/
    <scenario_id_0>.pkl
    <scenario_id_1>.pkl
    ...
```

Submit via Google Form before the deadline:

```text
Submission Google Form: TBD
Deadline: September 5, 2026 (AoE)
```

## Metrics

```text
Base =
0.30 Kinematic Realism
+ 0.35 Safety
+ 0.25 Cross-type Interaction

Coverage Bonus =
0.10 * Coverage * Kinematic Realism * Safety

Overall = Base + Coverage Bonus
```

See [docs/hetrod_metrics.md](docs/hetrod_metrics.md).

## Original WOSAC Tools

```bash
python prepare_gt.py /path/to/waymo/scenario/validation \
  --output-dir data/waymo_processed/validation_gt

python wosac_eval.py /path/to/rollout_dir \
  --gt-dir data/waymo_processed/validation_gt \
  --version 2025
```

## Tests

```bash
python -m unittest discover -s tests -p 'test_hetrod*.py'
```

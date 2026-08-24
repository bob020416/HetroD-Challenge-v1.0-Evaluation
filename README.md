# HetroD Challenge Evaluation Toolkit

Offline HetroD Challenge evaluator built on the fast WOSAC metric backend.
This repository contains the public `hetrod-0.8.0` metric contract used for
local validation. Test ground truth remains private and is evaluated with the
same code and configuration.

## Install

```bash
conda create -y -n wosac_eval python=3.11.9
conda activate wosac_eval
pip install -r requirements.txt
pip install --no-deps waymo-open-dataset-tf-2-12-0==1.6.7
python scripts/check_environment.py
```


## Data

Download the public dataset from the
[HetroD dataset page](https://levelxdata.com/hetrod-dataset/).

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

Validate an extracted public package:

```bash
python scripts/validate_public_dataset.py \
  --public-root /path/to/HetroD-Challenge-v1.0-public
```

The release archive is accompanied by a `.sha256` file. Verify it before
extracting with `sha256sum -c <archive-name>.sha256`.

## Validate Rollouts

Run inference on `valid/scenarionet`, write rollout pickles, then evaluate:

```bash
python hetrod_eval.py /path/to/valid_rollouts \
  --gt-dir /path/to/HetroD-Challenge-v1.0-public/valid/gt \
  --output valid_hetrod_metrics_report.json \
  --device cuda
```

## Submission

Official rollout file name:

```text
<scenario_id>.npz
```

Create it without object arrays:

```python
np.savez(
    output_path,
    agent_id=agent_id,                  # [num_agents]
    simulated_states=simulated_states,  # [32, num_agents, 80, 4]
)
```

The evaluator also accepts legacy `.pkl` dictionaries with the same two keys
for trusted local validation. Official organizer scoring rejects executable
pickle submissions by default.

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
    <scenario_id_0>.npz
    <scenario_id_1>.npz
    ...
```

Before uploading, run the public preflight against the extracted dataset:

```bash
python scripts/validate_submission.py your_team_submission.zip \
  --public-root /path/to/HetroD-Challenge-v1.0-public \
  --output submission_preflight.json
```

Preflight requires all 955 scenario files and checks exact required agent IDs,
32 rollouts, 80 future frames, `(x, y, z, yaw)`, floating-point dtype, and
finite values. It does not require or expose test GT.

The current submission portal, schedule, and challenge rules are published on
the [HetroD dataset page](https://levelxdata.com/hetrod-dataset/). Keeping that
information in one location avoids stale links or dates in evaluator releases.

## Metrics

The current challenge metric version is `hetrod-0.8.0`.

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
python -m unittest discover -s tests -p 'test_*.py'
```

## Organizer evaluation

The public evaluator uses the deterministic v0.8 reference-GT selector for
train/validation. Official test scoring may additionally load a versioned,
private selection/exclusion manifest; the manifest and test GT are never
committed. This keeps metric code identical while preventing leakage of
future-derived test targets.

```bash
python scripts/score_submission.py your_team_submission.zip \
  --public-root /path/to/HetroD-Challenge-v1.0-public \
  --private-gt-dir /private/path/test_gt \
  --selection-manifest /private/path/test_selection.json \
  --output /private/path/team_metrics.json \
  --device cuda
```

The command performs complete public-data preflight before loading private GT
or starting metric computation. See
[docs/organizer_evaluation.md](docs/organizer_evaluation.md).

> **Security:** pickle can execute arbitrary code. Organizer scoring must run
> each untrusted submission in a disposable, no-network worker with no
> credentials and no unrelated filesystem mounts.

## Release boundary

Only source code, documentation, tests, and small configuration files belong
in this repository. Dataset archives, ground-truth pickles, submissions,
reports, media, checkpoints, scheduler logs, and private test data are ignored
and must not be committed.

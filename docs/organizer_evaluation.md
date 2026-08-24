# Organizer evaluation runbook

This runbook intentionally contains no test GT, private target IDs, account
details, credentials, absolute home paths, or participant submissions.

## Frozen inputs

Keep these artifacts read-only and record their SHA-256 values with every run:

- public dataset release `schema1.3`;
- private test GT matching that release;
- private `hetrod-selection-manifest-v1` for metric `hetrod-0.8.0`;
- evaluator Git commit;
- original participant archive.

The private selection manifest contains all 955 test scenario IDs. Scored
records provide selected targets and optional explicit pairs; organizer
exclusions remain in the submission completeness check but are omitted from
the 946-scenario score aggregation.

Before accepting submissions, validate the frozen bundle once:

```bash
python scripts/validate_private_test_bundle.py \
  --public-root PUBLIC_ROOT \
  --private-gt-dir PRIVATE_TEST_GT \
  --selection-manifest PRIVATE_SELECTION.json
```

## Fast rejection

Run this before allocating a metric GPU:

```bash
python scripts/validate_submission.py TEAM.zip \
  --public-root PUBLIC_ROOT \
  --output TEAM_preflight.json
```

It rejects missing/extra/duplicate scenarios, wrong agent sets, wrong rollout
or horizon dimensions, non-floating states, and NaN/Inf.

## Official scoring

Official submissions should use numeric `.npz` files, which are loaded with
`allow_pickle=False`. Reject `.pkl` by default. If legacy pickle must be used,
load it only inside a disposable worker with networking disabled, no
credentials, and only these mounts:

- participant archive: read-only;
- public package: read-only;
- private GT: read-only;
- private selection manifest: read-only;
- one empty output directory: write-only for the job.

Then run:

```bash
python scripts/score_submission.py TEAM.zip \
  --public-root PUBLIC_ROOT \
  --private-gt-dir PRIVATE_TEST_GT \
  --selection-manifest PRIVATE_SELECTION.json \
  --output OUTPUT/TEAM_metrics.json \
  --device cuda \
  --work-dir NODE_LOCAL_SCRATCH
```

Accept the result only when:

- `preflight.status == "ok"`;
- `evaluation.summary.num_rollout_files == 955`;
- `evaluation.summary.num_gt_files == 955`;
- `evaluation.summary.num_successful_scenarios == 946`;
- `evaluation.summary.num_excluded_by_manifest == 9`;
- `evaluation.summary.num_errors == 0`;
- the reported selection-manifest SHA-256 matches the frozen artifact.

Archive the JSON report, source archive hash, evaluator commit, manifest hash,
and scheduler job log. Do not publish scenario-level test metrics.

## Rebuilding a private selection manifest

Export all Supabase annotation pages, then run the generic builder:

```bash
python scripts/build_selection_manifest.py \
  --curations ALL_CURATIONS.json \
  --automatic-selection AUTO_SELECTION.json \
  --exclusions ORGANIZER_EXCLUSIONS.json \
  --scenario-manifest PUBLIC_ROOT/manifests/test.txt \
  --gt-dir PRIVATE_TEST_GT \
  --sanitize-human-targets \
  --output PRIVATE_SELECTION.json
```

Without `--sanitize-human-targets`, the builder fails if any human target is
not an official required/eligible agent. Sanitized removals and pair changes
are retained in manifest audit metadata.

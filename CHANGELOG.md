# Changelog

All metric changes are versioned because leaderboard scores from different
metric versions are not directly comparable.

## 0.8.0 tooling hardening (metric unchanged)

- Add safe numeric `.npz` submissions and complete public preflight.
- Add versioned organizer-private target/exclusion manifest support.
- Add one-command organizer scoring and private bundle consistency checks.
- Fail fast on incompatible Waymo Open Dataset protobuf versions.

## 0.8.0

- Select up to four strong interaction pairs and eight scoring endpoints with
  explicit anchor/context roles and behavior-diverse ranking.
- Use strict new-collision detection relative to reference pair/frame overlap,
  without geometric collision tolerance.
- Score cross-type interaction only on explicitly selected cross-type pairs.
- Support type-specific polygon valid regions and transition-aware pedestrian
  core, crosswalk, and reference-supported road corridors.
- Restrict coverage to type-specific valid regions and keep it as a realism-
  and safety-gated bonus.
- Macro-average agent types, pair types, and locations to reduce frequency
  bias from common locations or vehicle-following scenarios.
- Validate prediction IDs, rollout count, tensor shape, and finite values
  before scoring.

The exact public contract is documented in
[`docs/hetrod_metrics.md`](docs/hetrod_metrics.md).

# Changelog

All metric changes are versioned because leaderboard scores from different
metric versions are not directly comparable.

## 0.8.0 submission and organizer tooling (metric unchanged)

- Preserve the originally published `.pkl` submission contract; optionally
  accept safe numeric `.npz` files with the same two payload keys.
- Add complete public submission preflight for directories and ZIP archives.
- Add versioned organizer-private target/exclusion manifest support.
- Define hidden-test targets as the frozen union of automatic v0.8 selection
  and valid human-curated additions.
- Add sequential and strictly merged parallel organizer scoring, plus private
  bundle consistency checks.
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

# Changelog

## v1.1.0

### Fixed

- **Case-ID collision in the condition experiment.** Probes were numbered from a fixed offset of
  100 after the ordinary benign cases. That was safe while groups held at most 100 cases and
  silently collided beyond it: at 150 cases per group, 50 probes shared an ID with the ordinary
  request they were derived from. Every structure keyed by case ID then merged each such pair, so
  both members were judged on the same evidence bundle and assigned the same detector flag.
  All 50 affected pairs returned identical answers and identical free-text reasons.

  Probes are now numbered after the ordinary cases, and `load_hard_negative_setup` asserts that
  case IDs are unique, so a collision fails loudly. `consolidate_results.py` refuses any result
  file whose case IDs collide.

  Affected, and re-run: the generated-probe condition runs at 150 per group for the three local
  models, with and without the anomaly score. The hosted model's run with the score was affected
  and could not be repeated; it has been removed. Its run without the score used 50 per group,
  where the old numbering never collided, and is unchanged.

  Not affected: the detector experiment (`run_scaled_experiment.py`), the hand-written probe runs,
  the graph-based W2 check and the multi-agent evaluation.

  Effect on reported figures: the reference detector's separation on the condition sample is
  -0.04, not +0.12 as v1.0.0 reported. Condition separations on probes change accordingly, since a
  third of the probe and ordinary groups had been judged on their partner's evidence.

### Added

- `run_subtype_check.py`: partitions attacks by how they depart from normal traffic, including an
  injection subset carrying explicit attack signatures, and rescores each partition.
- `run_verdict_control.py` and `analyse_verdict_control.py`: asks each model directly whether a
  request is an attack, on the same cases and evidence as the condition assessments.
- `run_robustness_check.py`: partitions attacks by structural departure from normal traffic.

## v1.0.0

Initial release accompanying the paper submission.

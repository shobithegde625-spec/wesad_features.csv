# Subject-held-out stress monitoring evaluation

Protocol: loso; subjects: 15; labeled windows: 445.

Outer-fold results evaluate the training procedure. The separately saved deployment bundle has its own provenance; these metrics are not a direct test of that fitted artifact.

| Method | Scenario | Coverage | All-window stress recall | Accepted stress F1 | Mean processing ms |
| --- | --- | --- | --- | --- | --- |
| legacy_logistic | clean | 1.000 | 0.706 | 0.657 | 1.817 |
| legacy_svm | clean | 1.000 | 0.756 | 0.768 | 1.071 |
| legacy_forest | clean | 1.000 | 0.806 | 0.719 | 11.939 |
| ecg | clean | 1.000 | 0.544 | 0.680 | 3.742 |
| confidence_only | clean | 0.948 | 0.506 | 0.681 | 3.675 |
| cascade | clean | 0.948 | 0.506 | 0.681 | 3.653 |
| no_abstention | clean | 1.000 | 0.544 | 0.680 | 3.652 |
| cascade_single_window | clean | 0.948 | 0.506 | 0.681 | 3.652 |

Accuracy/F1 describe accepted decisions. All-window stress recall counts abstentions as undetected stress. Probability metrics disclose their own probability coverage.

Participant bootstrap intervals, paired differences, split roles, reliability bins, routed reliability and confidence-ranked risk curves are saved beside this report.

Threshold sweeps are exploratory tradeoff curves, not independently retuned test operating points. Predeclared targets remain engineering targets; results do not establish clinical validity.

- 15-person laboratory data does not establish real-world alert rates
- Stored-feature timings exclude feature extraction
- Outer-fold estimates assess the training procedure, not the final deployment artifact

Multimodal corruption and physical-device energy targets remain unassessed unless the corresponding waveform and hardware experiments have been run.

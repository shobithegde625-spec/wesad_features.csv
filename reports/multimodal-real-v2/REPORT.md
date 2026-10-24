# Subject-held-out stress monitoring evaluation

Protocol: loso; subjects: 15; labeled windows: 434.

Outer-fold results evaluate the training procedure. The separately saved deployment bundle has its own provenance; these metrics are not a direct test of that fitted artifact.

| Method | Scenario | Coverage | All-window stress recall | Accepted stress F1 | Mean processing ms |
| --- | --- | --- | --- | --- | --- |
| legacy_logistic | clean | 1.000 | 0.704 | 0.641 | 5.423 |
| legacy_logistic | ecg_missing_burst | 0.000 | 0.000 | unavailable | 0.141 |
| legacy_logistic | ecg_flatline | 0.000 | 0.000 | unavailable | 0.203 |
| legacy_logistic | ecg_clipping | 1.000 | 0.632 | 0.441 | 5.042 |
| legacy_logistic | ecg_noise_moderate | 1.000 | 1.000 | 0.525 | 5.631 |
| legacy_logistic | ecg_noise_severe | 1.000 | 1.000 | 0.519 | 5.901 |
| legacy_logistic | ecg_amplitude_scale | 1.000 | 1.000 | 0.520 | 5.008 |
| legacy_logistic | eda_dropout | 1.000 | 0.704 | 0.641 | 4.995 |
| legacy_logistic | resp_dropout | 1.000 | 0.704 | 0.641 | 5.088 |
| legacy_logistic | alternate_dropout | 1.000 | 0.704 | 0.641 | 4.957 |
| legacy_logistic | total_dropout | 0.000 | 0.000 | unavailable | 0.128 |
| legacy_svm | clean | 1.000 | 0.743 | 0.756 | 4.663 |
| legacy_svm | ecg_missing_burst | 0.000 | 0.000 | unavailable | 0.139 |
| legacy_svm | ecg_flatline | 0.000 | 0.000 | unavailable | 0.204 |
| legacy_svm | ecg_clipping | 1.000 | 0.987 | 0.514 | 4.196 |
| legacy_svm | ecg_noise_moderate | 1.000 | 0.987 | 0.562 | 5.033 |
| legacy_svm | ecg_noise_severe | 1.000 | 1.000 | 0.519 | 4.942 |
| legacy_svm | ecg_amplitude_scale | 1.000 | 1.000 | 0.519 | 4.144 |
| legacy_svm | eda_dropout | 1.000 | 0.743 | 0.756 | 4.213 |
| legacy_svm | resp_dropout | 1.000 | 0.743 | 0.756 | 4.240 |
| legacy_svm | alternate_dropout | 1.000 | 0.743 | 0.756 | 4.169 |
| legacy_svm | total_dropout | 0.000 | 0.000 | unavailable | 0.126 |
| legacy_forest | clean | 1.000 | 0.809 | 0.721 | 15.329 |
| legacy_forest | ecg_missing_burst | 0.000 | 0.000 | unavailable | 0.138 |
| legacy_forest | ecg_flatline | 0.000 | 0.000 | unavailable | 0.202 |
| legacy_forest | ecg_clipping | 1.000 | 0.072 | 0.130 | 14.749 |
| legacy_forest | ecg_noise_moderate | 1.000 | 0.947 | 0.595 | 15.309 |
| legacy_forest | ecg_noise_severe | 1.000 | 1.000 | 0.519 | 15.378 |
| legacy_forest | ecg_amplitude_scale | 1.000 | 0.868 | 0.564 | 14.961 |
| legacy_forest | eda_dropout | 1.000 | 0.809 | 0.721 | 15.065 |
| legacy_forest | resp_dropout | 1.000 | 0.809 | 0.721 | 14.855 |
| legacy_forest | alternate_dropout | 1.000 | 0.809 | 0.721 | 14.809 |
| legacy_forest | total_dropout | 0.000 | 0.000 | unavailable | 0.125 |
| ecg | clean | 1.000 | 0.730 | 0.779 | 11.450 |
| confidence_only | clean | 1.000 | 0.882 | 0.810 | 14.463 |
| cascade | clean | 1.000 | 0.882 | 0.810 | 14.329 |
| no_abstention | clean | 1.000 | 0.882 | 0.810 | 14.459 |
| fusion | clean | 1.000 | 0.888 | 0.846 | 16.887 |
| cascade_single_window | clean | 1.000 | 0.882 | 0.810 | 14.519 |
| ecg | ecg_missing_burst | 0.000 | 0.000 | unavailable | 1.308 |
| confidence_only | ecg_missing_burst | 1.000 | 0.454 | 0.486 | 12.769 |
| cascade | ecg_missing_burst | 1.000 | 0.776 | 0.787 | 7.976 |
| no_abstention | ecg_missing_burst | 1.000 | 0.776 | 0.787 | 7.766 |
| fusion | ecg_missing_burst | 0.000 | 0.000 | unavailable | 1.442 |
| cascade_single_window | ecg_missing_burst | 1.000 | 0.776 | 0.787 | 8.395 |
| ecg | ecg_flatline | 0.000 | 0.000 | unavailable | 1.505 |
| confidence_only | ecg_flatline | 1.000 | 0.776 | 0.787 | 12.951 |
| cascade | ecg_flatline | 1.000 | 0.776 | 0.787 | 7.795 |
| no_abstention | ecg_flatline | 1.000 | 0.776 | 0.787 | 9.484 |
| fusion | ecg_flatline | 0.000 | 0.000 | unavailable | 1.362 |
| cascade_single_window | ecg_flatline | 1.000 | 0.776 | 0.787 | 7.942 |
| ecg | ecg_clipping | 0.567 | 0.092 | 0.173 | 7.821 |
| confidence_only | ecg_clipping | 1.000 | 0.579 | 0.522 | 13.433 |
| cascade | ecg_clipping | 1.000 | 0.704 | 0.639 | 11.865 |
| no_abstention | ecg_clipping | 1.000 | 0.704 | 0.639 | 11.838 |
| fusion | ecg_clipping | 0.567 | 0.467 | 0.516 | 11.033 |
| cascade_single_window | ecg_clipping | 1.000 | 0.704 | 0.639 | 11.861 |
| ecg | ecg_noise_moderate | 1.000 | 0.888 | 0.699 | 11.638 |
| confidence_only | ecg_noise_moderate | 1.000 | 0.947 | 0.696 | 15.451 |
| cascade | ecg_noise_moderate | 1.000 | 0.947 | 0.696 | 15.383 |
| no_abstention | ecg_noise_moderate | 1.000 | 0.947 | 0.696 | 15.485 |
| fusion | ecg_noise_moderate | 1.000 | 0.961 | 0.745 | 17.168 |
| cascade_single_window | ecg_noise_moderate | 1.000 | 0.947 | 0.696 | 15.419 |
| ecg | ecg_noise_severe | 0.000 | 0.000 | unavailable | 1.929 |
| confidence_only | ecg_noise_severe | 1.000 | 0.572 | 0.416 | 14.457 |
| cascade | ecg_noise_severe | 1.000 | 0.776 | 0.787 | 8.576 |
| no_abstention | ecg_noise_severe | 1.000 | 0.776 | 0.787 | 8.416 |
| fusion | ecg_noise_severe | 0.000 | 0.000 | unavailable | 1.935 |
| cascade_single_window | ecg_noise_severe | 1.000 | 0.776 | 0.787 | 8.337 |
| ecg | ecg_amplitude_scale | 1.000 | 0.441 | 0.427 | 10.899 |
| confidence_only | ecg_amplitude_scale | 1.000 | 0.539 | 0.471 | 12.184 |
| cascade | ecg_amplitude_scale | 1.000 | 0.539 | 0.471 | 12.628 |
| no_abstention | ecg_amplitude_scale | 1.000 | 0.539 | 0.471 | 12.178 |
| fusion | ecg_amplitude_scale | 1.000 | 0.934 | 0.606 | 16.602 |
| cascade_single_window | ecg_amplitude_scale | 1.000 | 0.539 | 0.471 | 12.195 |
| ecg | eda_dropout | 1.000 | 0.730 | 0.779 | 11.567 |
| confidence_only | eda_dropout | 1.000 | 0.730 | 0.779 | 11.026 |
| cascade | eda_dropout | 1.000 | 0.730 | 0.779 | 10.917 |
| no_abstention | eda_dropout | 1.000 | 0.730 | 0.779 | 10.886 |
| fusion | eda_dropout | 0.000 | 0.000 | unavailable | 8.585 |
| cascade_single_window | eda_dropout | 1.000 | 0.730 | 0.779 | 10.948 |
| ecg | resp_dropout | 1.000 | 0.730 | 0.779 | 10.469 |
| confidence_only | resp_dropout | 1.000 | 0.730 | 0.779 | 11.185 |
| cascade | resp_dropout | 1.000 | 0.730 | 0.779 | 11.223 |
| no_abstention | resp_dropout | 1.000 | 0.730 | 0.779 | 11.326 |
| fusion | resp_dropout | 0.000 | 0.000 | unavailable | 10.152 |
| cascade_single_window | resp_dropout | 1.000 | 0.730 | 0.779 | 11.223 |
| ecg | alternate_dropout | 1.000 | 0.730 | 0.779 | 10.179 |
| confidence_only | alternate_dropout | 1.000 | 0.730 | 0.779 | 10.114 |
| cascade | alternate_dropout | 1.000 | 0.730 | 0.779 | 9.938 |
| no_abstention | alternate_dropout | 1.000 | 0.730 | 0.779 | 12.969 |
| fusion | alternate_dropout | 0.000 | 0.000 | unavailable | 7.952 |
| cascade_single_window | alternate_dropout | 1.000 | 0.730 | 0.779 | 10.021 |
| ecg | total_dropout | 0.000 | 0.000 | unavailable | 0.021 |
| confidence_only | total_dropout | 0.000 | 0.000 | unavailable | 0.028 |
| cascade | total_dropout | 0.000 | 0.000 | unavailable | 0.020 |
| no_abstention | total_dropout | 0.000 | 0.000 | unavailable | 0.020 |
| fusion | total_dropout | 0.000 | 0.000 | unavailable | 0.020 |
| cascade_single_window | total_dropout | 0.000 | 0.000 | unavailable | 0.020 |

Accuracy/F1 describe accepted decisions. All-window stress recall counts abstentions as undetected stress. Probability metrics disclose their own probability coverage.

Participant bootstrap intervals, paired differences, split roles, reliability bins, routed reliability and confidence-ranked risk curves are saved beside this report.

Threshold sweeps are exploratory tradeoff curves, not independently retuned test operating points. Predeclared targets remain engineering targets; results do not establish clinical validity.

- 15-person laboratory data does not establish real-world alert rates
- Energy and sensor acquisition savings are not measured
- Outer-fold estimates assess the training procedure, not the final deployment artifact

Multimodal corruption and physical-device energy targets remain unassessed unless the corresponding waveform and hardware experiments have been run.

Read [the supported research claim and limitations](../../RESEARCH_CLAIM.md) and [the verified before/after comparison](../multimodal-revision-comparison/FINDINGS.md) before interpreting the revised result.

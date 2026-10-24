# Subject-held-out stress monitoring evaluation

Protocol: loso; subjects: 15; labeled windows: 434.

Outer-fold results evaluate the training procedure. The separately saved deployment bundle has its own provenance; these metrics are not a direct test of that fitted artifact.

| Method | Scenario | Coverage | All-window stress recall | Accepted stress F1 | Mean processing ms |
| --- | --- | --- | --- | --- | --- |
| legacy_logistic | clean | 1.000 | 0.704 | 0.641 | 5.268 |
| legacy_logistic | ecg_missing_burst | 0.000 | 0.000 | unavailable | 0.163 |
| legacy_logistic | ecg_flatline | 0.000 | 0.000 | unavailable | 0.233 |
| legacy_logistic | ecg_clipping | 1.000 | 0.632 | 0.441 | 5.429 |
| legacy_logistic | ecg_noise_moderate | 1.000 | 1.000 | 0.525 | 6.495 |
| legacy_logistic | ecg_noise_severe | 1.000 | 1.000 | 0.519 | 6.204 |
| legacy_logistic | ecg_amplitude_scale | 1.000 | 1.000 | 0.520 | 5.226 |
| legacy_logistic | eda_dropout | 1.000 | 0.704 | 0.641 | 5.106 |
| legacy_logistic | resp_dropout | 1.000 | 0.704 | 0.641 | 5.077 |
| legacy_logistic | alternate_dropout | 1.000 | 0.704 | 0.641 | 4.994 |
| legacy_logistic | total_dropout | 0.000 | 0.000 | unavailable | 0.132 |
| legacy_svm | clean | 1.000 | 0.743 | 0.756 | 4.251 |
| legacy_svm | ecg_missing_burst | 0.000 | 0.000 | unavailable | 0.143 |
| legacy_svm | ecg_flatline | 0.000 | 0.000 | unavailable | 0.212 |
| legacy_svm | ecg_clipping | 1.000 | 0.987 | 0.514 | 4.606 |
| legacy_svm | ecg_noise_moderate | 1.000 | 0.987 | 0.562 | 5.323 |
| legacy_svm | ecg_noise_severe | 1.000 | 1.000 | 0.519 | 5.150 |
| legacy_svm | ecg_amplitude_scale | 1.000 | 1.000 | 0.519 | 4.456 |
| legacy_svm | eda_dropout | 1.000 | 0.743 | 0.756 | 4.217 |
| legacy_svm | resp_dropout | 1.000 | 0.743 | 0.756 | 4.344 |
| legacy_svm | alternate_dropout | 1.000 | 0.743 | 0.756 | 4.328 |
| legacy_svm | total_dropout | 0.000 | 0.000 | unavailable | 0.128 |
| legacy_forest | clean | 1.000 | 0.809 | 0.721 | 15.360 |
| legacy_forest | ecg_missing_burst | 0.000 | 0.000 | unavailable | 0.139 |
| legacy_forest | ecg_flatline | 0.000 | 0.000 | unavailable | 0.209 |
| legacy_forest | ecg_clipping | 1.000 | 0.072 | 0.130 | 14.800 |
| legacy_forest | ecg_noise_moderate | 1.000 | 0.947 | 0.595 | 15.657 |
| legacy_forest | ecg_noise_severe | 1.000 | 1.000 | 0.519 | 15.399 |
| legacy_forest | ecg_amplitude_scale | 1.000 | 0.868 | 0.564 | 14.745 |
| legacy_forest | eda_dropout | 1.000 | 0.809 | 0.721 | 15.202 |
| legacy_forest | resp_dropout | 1.000 | 0.809 | 0.721 | 14.722 |
| legacy_forest | alternate_dropout | 1.000 | 0.809 | 0.721 | 14.617 |
| legacy_forest | total_dropout | 0.000 | 0.000 | unavailable | 0.125 |
| ecg | clean | 1.000 | 0.803 | 0.765 | 11.705 |
| confidence_only | clean | 0.915 | 0.842 | 0.831 | 12.529 |
| cascade | clean | 0.915 | 0.842 | 0.831 | 12.482 |
| no_abstention | clean | 1.000 | 0.849 | 0.814 | 12.567 |
| fusion | clean | 1.000 | 0.888 | 0.846 | 17.185 |
| cascade_single_window | clean | 0.915 | 0.842 | 0.831 | 12.801 |
| ecg | ecg_missing_burst | 0.000 | 0.000 | unavailable | 1.320 |
| confidence_only | ecg_missing_burst | 0.982 | 0.730 | 0.583 | 12.911 |
| cascade | ecg_missing_burst | 0.919 | 0.678 | 0.752 | 9.201 |
| no_abstention | ecg_missing_burst | 1.000 | 0.717 | 0.722 | 9.762 |
| fusion | ecg_missing_burst | 0.000 | 0.000 | unavailable | 1.345 |
| cascade_single_window | ecg_missing_burst | 0.919 | 0.678 | 0.752 | 10.926 |
| ecg | ecg_flatline | 0.000 | 0.000 | unavailable | 1.357 |
| confidence_only | ecg_flatline | 0.919 | 0.678 | 0.752 | 14.041 |
| cascade | ecg_flatline | 0.919 | 0.678 | 0.752 | 9.076 |
| no_abstention | ecg_flatline | 1.000 | 0.717 | 0.722 | 9.059 |
| fusion | ecg_flatline | 0.000 | 0.000 | unavailable | 1.360 |
| cascade_single_window | ecg_flatline | 0.919 | 0.678 | 0.752 | 8.985 |
| ecg | ecg_clipping | 0.567 | 0.211 | 0.242 | 7.741 |
| confidence_only | ecg_clipping | 0.991 | 0.704 | 0.466 | 13.637 |
| cascade | ecg_clipping | 0.961 | 0.612 | 0.446 | 12.047 |
| no_abstention | ecg_clipping | 1.000 | 0.638 | 0.447 | 12.009 |
| fusion | ecg_clipping | 0.567 | 0.467 | 0.516 | 10.703 |
| cascade_single_window | ecg_clipping | 0.961 | 0.612 | 0.446 | 12.015 |
| ecg | ecg_noise_moderate | 1.000 | 0.862 | 0.567 | 11.685 |
| confidence_only | ecg_noise_moderate | 0.945 | 0.914 | 0.610 | 13.195 |
| cascade | ecg_noise_moderate | 0.945 | 0.914 | 0.610 | 13.269 |
| no_abstention | ecg_noise_moderate | 1.000 | 0.914 | 0.594 | 13.079 |
| fusion | ecg_noise_moderate | 1.000 | 0.961 | 0.739 | 16.700 |
| cascade_single_window | ecg_noise_moderate | 0.945 | 0.914 | 0.610 | 13.241 |
| ecg | ecg_noise_severe | 1.000 | 0.868 | 0.500 | 11.893 |
| confidence_only | ecg_noise_severe | 1.000 | 0.934 | 0.510 | 12.646 |
| cascade | ecg_noise_severe | 1.000 | 0.934 | 0.510 | 12.730 |
| no_abstention | ecg_noise_severe | 1.000 | 0.934 | 0.510 | 12.893 |
| fusion | ecg_noise_severe | 1.000 | 0.967 | 0.579 | 16.931 |
| cascade_single_window | ecg_noise_severe | 1.000 | 0.934 | 0.510 | 12.686 |
| ecg | ecg_amplitude_scale | 1.000 | 0.849 | 0.535 | 11.076 |
| confidence_only | ecg_amplitude_scale | 0.995 | 0.914 | 0.542 | 11.894 |
| cascade | ecg_amplitude_scale | 0.995 | 0.914 | 0.542 | 11.827 |
| no_abstention | ecg_amplitude_scale | 1.000 | 0.914 | 0.542 | 11.827 |
| fusion | ecg_amplitude_scale | 1.000 | 0.934 | 0.606 | 15.952 |
| cascade_single_window | ecg_amplitude_scale | 0.995 | 0.914 | 0.542 | 11.876 |
| ecg | eda_dropout | 1.000 | 0.803 | 0.765 | 10.617 |
| confidence_only | eda_dropout | 0.859 | 0.783 | 0.826 | 10.599 |
| cascade | eda_dropout | 0.859 | 0.783 | 0.826 | 10.654 |
| no_abstention | eda_dropout | 1.000 | 0.803 | 0.765 | 10.543 |
| fusion | eda_dropout | 0.000 | 0.000 | unavailable | 7.702 |
| cascade_single_window | eda_dropout | 0.859 | 0.783 | 0.826 | 10.653 |
| ecg | resp_dropout | 1.000 | 0.803 | 0.765 | 10.526 |
| confidence_only | resp_dropout | 0.859 | 0.783 | 0.826 | 10.706 |
| cascade | resp_dropout | 0.859 | 0.783 | 0.826 | 10.745 |
| no_abstention | resp_dropout | 1.000 | 0.803 | 0.765 | 10.661 |
| fusion | resp_dropout | 0.000 | 0.000 | unavailable | 9.462 |
| cascade_single_window | resp_dropout | 0.859 | 0.783 | 0.826 | 10.669 |
| ecg | alternate_dropout | 1.000 | 0.803 | 0.765 | 11.361 |
| confidence_only | alternate_dropout | 0.859 | 0.783 | 0.826 | 11.041 |
| cascade | alternate_dropout | 0.859 | 0.783 | 0.826 | 11.731 |
| no_abstention | alternate_dropout | 1.000 | 0.803 | 0.765 | 12.640 |
| fusion | alternate_dropout | 0.000 | 0.000 | unavailable | 7.604 |
| cascade_single_window | alternate_dropout | 0.859 | 0.783 | 0.826 | 11.517 |
| ecg | total_dropout | 0.000 | 0.000 | unavailable | 0.026 |
| confidence_only | total_dropout | 0.000 | 0.000 | unavailable | 0.039 |
| cascade | total_dropout | 0.000 | 0.000 | unavailable | 0.027 |
| no_abstention | total_dropout | 0.000 | 0.000 | unavailable | 0.024 |
| fusion | total_dropout | 0.000 | 0.000 | unavailable | 0.023 |
| cascade_single_window | total_dropout | 0.000 | 0.000 | unavailable | 0.028 |

Accuracy/F1 describe accepted decisions. All-window stress recall counts abstentions as undetected stress. Probability metrics disclose their own probability coverage.

Participant bootstrap intervals, paired differences, split roles, reliability bins, routed reliability and confidence-ranked risk curves are saved beside this report.

Threshold sweeps are exploratory tradeoff curves, not independently retuned test operating points. Predeclared targets remain engineering targets; results do not establish clinical validity.

- 15-person laboratory data does not establish real-world alert rates
- Energy and sensor acquisition savings are not measured
- Outer-fold estimates assess the training procedure, not the final deployment artifact

Multimodal corruption and physical-device energy targets remain unassessed unless the corresponding waveform and hardware experiments have been run.

Read [FINDINGS.md](FINDINGS.md) for target failures, temporal metric definitions, active-event exposure and the dedicated artifact benchmark.

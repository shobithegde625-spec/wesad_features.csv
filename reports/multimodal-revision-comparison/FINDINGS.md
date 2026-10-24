# Revised-model comparison

Both versions use identical source windows, target labels, scenarios and outer participant splits. This is an exploratory revision on an already observed cohort, not independent confirmation.

| Version | Method | Scenario | Coverage | All-window stress recall | Accepted stress F1 | Mean CPU ms |
| --- | --- | --- | --- | --- | --- | --- |
| v1 | legacy_svm | clean | 1.0000 | 0.7434 | 0.7559 | unavailable |
| v1 | legacy_svm | ecg_missing_burst | 0.0000 | 0.0000 | unavailable | unavailable |
| v1 | legacy_svm | ecg_flatline | 0.0000 | 0.0000 | unavailable | unavailable |
| v1 | legacy_svm | ecg_clipping | 1.0000 | 0.9868 | 0.5137 | unavailable |
| v1 | legacy_svm | ecg_noise_moderate | 1.0000 | 0.9868 | 0.5618 | unavailable |
| v1 | legacy_svm | ecg_noise_severe | 1.0000 | 1.0000 | 0.5188 | unavailable |
| v1 | legacy_svm | ecg_amplitude_scale | 1.0000 | 1.0000 | 0.5188 | unavailable |
| v1 | legacy_svm | eda_dropout | 1.0000 | 0.7434 | 0.7559 | unavailable |
| v1 | legacy_svm | resp_dropout | 1.0000 | 0.7434 | 0.7559 | unavailable |
| v1 | legacy_svm | alternate_dropout | 1.0000 | 0.7434 | 0.7559 | unavailable |
| v1 | legacy_svm | total_dropout | 0.0000 | 0.0000 | unavailable | unavailable |
| v1 | ecg | clean | 1.0000 | 0.8026 | 0.7649 | 11.3245 |
| v1 | cascade | clean | 0.9147 | 0.8421 | 0.8312 | 12.2561 |
| v1 | fusion | clean | 1.0000 | 0.8882 | 0.8464 | 16.2761 |
| v1 | ecg | ecg_missing_burst | 0.0000 | 0.0000 | unavailable | 1.3047 |
| v1 | cascade | ecg_missing_burst | 0.9194 | 0.6776 | 0.7518 | 9.0760 |
| v1 | fusion | ecg_missing_burst | 0.0000 | 0.0000 | unavailable | 1.3072 |
| v1 | ecg | ecg_flatline | 0.0000 | 0.0000 | unavailable | 1.3501 |
| v1 | cascade | ecg_flatline | 0.9194 | 0.6776 | 0.7518 | 8.9722 |
| v1 | fusion | ecg_flatline | 0.0000 | 0.0000 | unavailable | 1.3513 |
| v1 | ecg | ecg_clipping | 0.5668 | 0.2105 | 0.2424 | 7.6876 |
| v1 | cascade | ecg_clipping | 0.9608 | 0.6118 | 0.4460 | 11.9452 |
| v1 | fusion | ecg_clipping | 0.5668 | 0.4671 | 0.5164 | 10.6179 |
| v1 | ecg | ecg_noise_moderate | 1.0000 | 0.8618 | 0.5671 | 11.6078 |
| v1 | cascade | ecg_noise_moderate | 0.9447 | 0.9145 | 0.6096 | 13.1334 |
| v1 | fusion | ecg_noise_moderate | 1.0000 | 0.9605 | 0.7392 | 16.5660 |
| v1 | ecg | ecg_noise_severe | 1.0000 | 0.8684 | 0.5000 | 11.7585 |
| v1 | cascade | ecg_noise_severe | 1.0000 | 0.9342 | 0.5099 | 12.5724 |
| v1 | fusion | ecg_noise_severe | 1.0000 | 0.9671 | 0.5787 | 16.7166 |
| v1 | ecg | ecg_amplitude_scale | 1.0000 | 0.8487 | 0.5353 | 10.9777 |
| v1 | cascade | ecg_amplitude_scale | 0.9954 | 0.9145 | 0.5419 | 11.7645 |
| v1 | fusion | ecg_amplitude_scale | 1.0000 | 0.9342 | 0.6055 | 15.8608 |
| v1 | ecg | eda_dropout | 1.0000 | 0.8026 | 0.7649 | 10.5260 |
| v1 | cascade | eda_dropout | 0.8594 | 0.7829 | 0.8264 | 10.4923 |
| v1 | fusion | eda_dropout | 0.0000 | 0.0000 | unavailable | 7.6401 |
| v1 | ecg | resp_dropout | 1.0000 | 0.8026 | 0.7649 | 10.4082 |
| v1 | cascade | resp_dropout | 0.8594 | 0.7829 | 0.8264 | 10.6586 |
| v1 | fusion | resp_dropout | 0.0000 | 0.0000 | unavailable | 9.4181 |
| v1 | ecg | alternate_dropout | 1.0000 | 0.8026 | 0.7649 | 10.3585 |
| v1 | cascade | alternate_dropout | 0.8594 | 0.7829 | 0.8264 | 10.3381 |
| v1 | fusion | alternate_dropout | 0.0000 | 0.0000 | unavailable | 7.2994 |
| v1 | ecg | total_dropout | 0.0000 | 0.0000 | unavailable | 0.0244 |
| v1 | cascade | total_dropout | 0.0000 | 0.0000 | unavailable | 0.0230 |
| v1 | fusion | total_dropout | 0.0000 | 0.0000 | unavailable | 0.0218 |
| v2 | legacy_svm | clean | 1.0000 | 0.7434 | 0.7559 | unavailable |
| v2 | legacy_svm | ecg_missing_burst | 0.0000 | 0.0000 | unavailable | unavailable |
| v2 | legacy_svm | ecg_flatline | 0.0000 | 0.0000 | unavailable | unavailable |
| v2 | legacy_svm | ecg_clipping | 1.0000 | 0.9868 | 0.5137 | unavailable |
| v2 | legacy_svm | ecg_noise_moderate | 1.0000 | 0.9868 | 0.5618 | unavailable |
| v2 | legacy_svm | ecg_noise_severe | 1.0000 | 1.0000 | 0.5188 | unavailable |
| v2 | legacy_svm | ecg_amplitude_scale | 1.0000 | 1.0000 | 0.5188 | unavailable |
| v2 | legacy_svm | eda_dropout | 1.0000 | 0.7434 | 0.7559 | unavailable |
| v2 | legacy_svm | resp_dropout | 1.0000 | 0.7434 | 0.7559 | unavailable |
| v2 | legacy_svm | alternate_dropout | 1.0000 | 0.7434 | 0.7559 | unavailable |
| v2 | legacy_svm | total_dropout | 0.0000 | 0.0000 | unavailable | unavailable |
| v2 | ecg | clean | 1.0000 | 0.7303 | 0.7789 | 11.2541 |
| v2 | cascade | clean | 1.0000 | 0.8816 | 0.8097 | 14.2426 |
| v2 | fusion | clean | 1.0000 | 0.8882 | 0.8464 | 16.7719 |
| v2 | ecg | ecg_missing_burst | 0.0000 | 0.0000 | unavailable | 1.2980 |
| v2 | cascade | ecg_missing_burst | 1.0000 | 0.7763 | 0.7867 | 7.7900 |
| v2 | fusion | ecg_missing_burst | 0.0000 | 0.0000 | unavailable | 1.4269 |
| v2 | ecg | ecg_flatline | 0.0000 | 0.0000 | unavailable | 1.4995 |
| v2 | cascade | ecg_flatline | 1.0000 | 0.7763 | 0.7867 | 7.7820 |
| v2 | fusion | ecg_flatline | 0.0000 | 0.0000 | unavailable | 1.3599 |
| v2 | ecg | ecg_clipping | 0.5668 | 0.0921 | 0.1728 | 7.8019 |
| v2 | cascade | ecg_clipping | 1.0000 | 0.7039 | 0.6388 | 11.8356 |
| v2 | fusion | ecg_clipping | 0.5668 | 0.4671 | 0.5164 | 10.9956 |
| v2 | ecg | ecg_noise_moderate | 1.0000 | 0.8882 | 0.6995 | 11.5621 |
| v2 | cascade | ecg_noise_moderate | 1.0000 | 0.9474 | 0.6957 | 15.3057 |
| v2 | fusion | ecg_noise_moderate | 1.0000 | 0.9605 | 0.7449 | 17.0627 |
| v2 | ecg | ecg_noise_severe | 0.0000 | 0.0000 | unavailable | 1.9226 |
| v2 | cascade | ecg_noise_severe | 1.0000 | 0.7763 | 0.7867 | 8.3881 |
| v2 | fusion | ecg_noise_severe | 0.0000 | 0.0000 | unavailable | 1.9280 |
| v2 | ecg | ecg_amplitude_scale | 1.0000 | 0.4408 | 0.4268 | 10.8485 |
| v2 | cascade | ecg_amplitude_scale | 1.0000 | 0.5395 | 0.4713 | 12.4212 |
| v2 | fusion | ecg_amplitude_scale | 1.0000 | 0.9342 | 0.6055 | 16.5201 |
| v2 | ecg | eda_dropout | 1.0000 | 0.7303 | 0.7789 | 10.6099 |
| v2 | cascade | eda_dropout | 1.0000 | 0.7303 | 0.7789 | 10.6054 |
| v2 | fusion | eda_dropout | 0.0000 | 0.0000 | unavailable | 8.3623 |
| v2 | ecg | resp_dropout | 1.0000 | 0.7303 | 0.7789 | 10.4061 |
| v2 | cascade | resp_dropout | 1.0000 | 0.7303 | 0.7789 | 11.1138 |
| v2 | fusion | resp_dropout | 0.0000 | 0.0000 | unavailable | 10.0556 |
| v2 | ecg | alternate_dropout | 1.0000 | 0.7303 | 0.7789 | 9.9620 |
| v2 | cascade | alternate_dropout | 1.0000 | 0.7303 | 0.7789 | 9.9011 |
| v2 | fusion | alternate_dropout | 0.0000 | 0.0000 | unavailable | 7.7297 |
| v2 | ecg | total_dropout | 0.0000 | 0.0000 | unavailable | 0.0209 |
| v2 | cascade | total_dropout | 0.0000 | 0.0000 | unavailable | 0.0202 |
| v2 | fusion | total_dropout | 0.0000 | 0.0000 | unavailable | 0.0196 |

## Unchanged combined target

- v1: not_met_on_this_host; CPU fraction saved 0.24698580187341335; recall loss 0.046052631578947456.
- v2: not_met_on_this_host; CPU fraction saved 0.15080578176920212; recall loss 0.006578947368421129.

CPU comparisons are paired against each version's fusion expert on that run's host. Different runs can have different background loads; cross-version absolute timing is descriptive. The dedicated bundle benchmark measures a different fitted artifact and must not be combined with these recall estimates as a deployment guarantee.

Participant-paired bootstrap intervals are in paired_revision_differences.json. Intervals quantify participant variability within this cohort and do not remove adaptive reuse of the dataset. Accepted-only F1 must be read alongside coverage and all-window recall. A failed quality check is an unknown measurement, never evidence of baseline.

No battery, clinical or physical edge-device result is implied. The complete per-version reports retain event onset and delay limitations.

The separate final-artifact benchmark is cost-neutral on its sampled clean workload (cascade 19.83 ms versus fusion 19.65 ms CPU). Read [the full research claim and limitations](../../RESEARCH_CLAIM.md), including amplitude-scaling regressions and uncertainty.

# Real WESAD experiment findings

Official archive SHA-256: `5e15d2606adf16d819ba535c1786d92e94bae5ff5392f7ae01fb63f9938fd71c`. The 445 original classification windows were reproduced before fitting the multimodal experts. The chronological replay contains 1,442 windows; 434 pure baseline/stress windows enter binary scoring.

## Held-out results

The complete experiment uses 15 outer subject-held-out folds and eleven clean/corruption scenarios. Fitting, calibration and policy-selection participants remain disjoint from each outer subject.

| Clean method | Accepted coverage | All-window stress recall | Accepted stress F1 |
| --- | ---: | ---: | ---: |
| Original-feature SVM | 100.0% | 74.3% | 0.756 |
| Quality-aware cascade | 91.5% | 84.2% | 0.831 |
| Always-on fusion | 100.0% | 88.8% | 0.846 |

The cascade saved about 24.7% of mean processing CPU time relative to fusion on eligible clean windows, but stress recall fell by 4.6 percentage points. The predeclared combined target (at least 25% CPU saving, at most 3-point recall loss) was not met. These are point estimates; participant-level intervals and paired differences are in the main report.

With ECG flatline, the cascade retains 91.9% coverage and 67.8% all-window stress recall through the alternate expert; static full-modality fusion cannot produce decisions. Under severe ECG noise, the cascade accepts every scored window but its stress F1 falls to 0.510. Confidence and the current quality heuristics therefore do not provide adequate noise rejection. Total dropout correctly returns insufficient evidence for every window.

## Temporal interpretation

The event_recall field in the main report counts new event starts inside eligible stress episodes. It does not credit an event that was already active before that episode. Onset delay begins at the first fully labeled stress-window start, rather than the exact protocol transition; mixed windows are unscored.

Temporal confirmation reduces baseline event starts from ten to four, but only three of fifteen stress episodes contain a new cascade event start. This is not evidence of a reliable episode-onset monitor. A low number of new false starts can also coexist with an event remaining active during baseline; the additional exposure review below makes that visible.

| Method | Eligible baseline time with an active event | Stress blocks with any active event |
| --- | ---: | ---: |
| cascade | 7.8% | 14/15 |
| cascade_single_window | 10.3% | 14/15 |
| fusion | 2.8% | 13/15 |

Any-active coverage is descriptive exposure, not successful onset detection. It may include events opened during unsupported conditions. See event_exposure_review.json for exact quantities.

## Artifact-specific host benchmark

This separate benchmark uses the final fitted artifact `180f6c507c3a`, 100 source windows, three repeats and three scenarios. Every policy runs feature extraction afresh, and policy order rotates. These host timings cover all sampled conditions and a different fitted artifact from the outer folds; they should not be combined with the held-out recall figures as a single deployment guarantee.

| Mode | Scenario | Mean CPU ms | p95 processing ms |
| --- | --- | ---: | ---: |
| cascade | clean | 11.085 | 20.448 |
| cascade | ecg_flatline | 11.786 | 13.324 |
| cascade | total_dropout | 0.022 | 0.031 |
| ecg | clean | 10.074 | 11.933 |
| ecg | ecg_flatline | 1.360 | 1.474 |
| ecg | total_dropout | 0.022 | 0.027 |
| fusion | clean | 19.235 | 23.018 |
| fusion | ecg_flatline | 1.344 | 1.480 |
| fusion | total_dropout | 0.022 | 0.026 |

Processing excludes file IO, acquisition/window duration and sensor/radio energy. Process RSS also includes the offline participant cache and allocator retention. This is a host benchmark, not a physical edge-device or battery measurement.

## Conclusion

The implementation supports real multimodal inference, graceful ECG-failure routing and explicit rejection. The evidence does not establish the full reliability/compute target. Stronger noise-quality validation and temporal-onset evaluation are needed before field use. Additional real-world recordings, physical target-device testing and power instrumentation remain outstanding.

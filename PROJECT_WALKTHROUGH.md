# What changed, in plain language

The original project classified stored ECG measurements as baseline or stress. The current project also decides which sensors to trust, when to request more information, and when the measurements cannot support an answer. It includes a reproducible experiment and a paper describing both the gains and the limitations.

## Original project versus current project

| Area | Original code | Current implementation |
| --- | --- | --- |
| Sensor information | Seven ECG amplitude statistics from a feature CSV | Original ECG statistics plus heart/beat information, skin conductance (EDA) and respiration features |
| Prediction | A saved ECG Random Forest and separate model-comparison scripts | Independently trained ECG, EDA/respiration and full-fusion experts |
| Choosing sensors | A fixed ECG input | ECG first; request fusion when needed; use the alternate expert when ECG fails |
| Damaged measurements | No complete multimodal quality/fallback pipeline | Missing, flatline, clipping, beat/cycle validity and high-frequency ECG checks; explicit insufficient evidence |
| Confidence | Raw classifier probabilities | A separate participant group fits a probability calibrator; confidence still has uncertainty |
| Model selection | Original logistic/SVM/forest comparisons | Participant-weighted grouped selection, tuned SVM candidates and a compact forest candidate |
| Decisions over time | CSV playback driven by a UI timer | Recording timestamps, separate sustained-event state, resets at gaps/participant changes/restarts, stop at completion |
| Evaluation | Existing grouped holdout/five-fold comparison | Fifteen outer leave-one-subject-out folds, internal fit/calibration/policy roles, eleven waveform scenarios and ablations |
| Efficiency | Mainly model prediction measurements | Quality checks, waveform feature extraction, inference, routing and event processing measured together |
| Dashboard | Basic stress demonstration | Sensor quality, chosen expert, rejection reason, source time, artifact identity, replay and exported history |
| Research report | An Evaluation 1 proposal | A compiled Evaluation 2 paper with original diagrams, real screenshots, exact result tables and references |
| Code organization | Much of the logic repeated across scripts | Shared `pulsepoint/` modules called by the existing CLI scripts and dashboard |

The original code already used participant grouping in parts of its evaluation. The revision extends that separation to model selection, calibration and routing; it does not claim that all prior evaluation was ungrouped.

## What happens for one prediction

1. Read a completed 60-second interval from the recording.
2. Check whether its sensor signals are present and usable.
3. Extract ECG features and ask the ECG expert for a calibrated stress probability.
4. If ECG meets its early-exit confidence rule, accept that result without running fusion.
5. If ECG needs more evidence, combine it with skin conductance and breathing. If those channels are unavailable, usable ECG can still pass a separate final acceptance rule, with the reason recorded.
6. If ECG has failed, use the independently trained skin-conductance/breathing expert.
7. If no supported expert has enough usable evidence, return insufficient evidence. Missing data does not become a baseline prediction.
8. Record the decision and its source interval. During replay, update the separate temporal event state.

ECG measures electrical activity associated with heartbeats. EDA measures changes in skin conductance. Respiration measures breathing. The alternate expert is trained to work without ECG, rather than filling in made-up ECG values.

## What the data and training changes mean

The public WESAD archive was downloaded and its official checksum checked. The pipeline can read its compressed recordings directly, avoiding a large unpacked copy. We preserved the original data and recreated its 445 training windows from the raw recordings. A second, chronological replay table contains 1,442 windows. Only 434 have a complete baseline or stress label; mixed/other conditions remain unscored for binary accuracy.

In each outer experiment, one person's recordings are held out. The other participants provide separate groups for learning the experts, calibrating confidence and choosing routing rules. The held-out person's labels do not decide which expert runs. Each of the fifteen people becomes the held-out participant once.

The v2 revision separates two questions: “Should I request more sensors?” and “Can I accept the final answer?” It also adds SVM parameter candidates, a smaller forest and an ECG noise check. These changes improve some outcomes but do not solve every failure mode.

## The three main observations

1. **More stress detections than the original-feature ECG SVM.** On the same clean held-out windows, stress recall rises from 74.34% to 88.16%; accuracy rises from 83.18% to 85.48%, and F1 from 0.756 to 0.810. The cascade makes more false stress predictions too: 45 versus 34.
2. **A measured cost/recall tradeoff across the held-out procedure.** The cascade uses 15.08% less mean CPU than its fold-specific fusion reference, with stress recall 0.66 percentage points lower. It does not achieve the original 25% saving target.
3. **Useful operation under ECG failure.** With ECG flatlined, the alternate expert retains 85.25% accuracy, 77.63% stress recall and 0.787 F1 at full coverage. Severe-noise F1 rises from 0.510 in v1 to 0.787 in v2, with lower stress recall but fewer false stress predictions.

Recall means the fraction of actual stress windows detected. Precision means the fraction of stress predictions that are correct. F1 combines precision and recall. Coverage means the fraction of eligible windows receiving an accepted answer. We also report all-window recall so rejecting difficult stress windows cannot make detection appear better.

## Why some results remain mixed

The final fitted dashboard artifact escalates to fusion on 83% of its representative clean benchmark windows. Its cascade takes 19.83 ms mean CPU versus fusion's 19.65 ms, so that artifact has no demonstrated saving. The held-out procedure's 15.08% saving must not be assigned to the final fitted model.

Amplitude-dependent ECG features can give confident wrong answers after sensor-gain changes. Multiplying amplitude by three reduces v2 cascade stress recall to 53.95%. Pooled clean F1 is also below v1's (0.810 versus 0.831), and alternate dropout reduces recall despite full coverage. With only fifteen people, calibration and policy estimates vary substantially across groups. Event-onset detection remains weak. These are recorded research findings, not hidden exceptions.

The system is a local prerecorded research prototype. We did not implement a live wearable link, train a CNN, deploy to a physical edge device, measure battery energy or demonstrate clinical validity. Adaptive sensing has prior art; the contribution here is the particular integrated implementation and reproducible evaluation.

## Where the work lives

| File or folder | Purpose |
| --- | --- |
| `pulsepoint/features.py` | Shared physiological features, lazy extraction and quality checks |
| `pulsepoint/models.py`, `training.py` | Expert selection, participant roles, calibration and versioned bundles |
| `pulsepoint/routing.py` | ECG/fusion/alternate selection, confidence rules and rejection |
| `pulsepoint/temporal.py`, `replay.py` | Source-time event state and shared inference runtime |
| `pulsepoint/evaluation.py`, `corruption.py`, `metrics.py` | Held-out comparisons, reproducible sensor failures and metrics |
| `pulsepoint/benchmark.py` | Independent processing measurements |
| `download_wesad.py`, `import_wesad.py`, `prepare_dataset.py` | Verified acquisition, archive handling and dataset preparation |
| `app.py` | Updated Streamlit dashboard |
| `tests/` | Forty-one automated tests, including the dashboard and end-to-end synthetic pipeline |
| `reports/` | Reproducible summaries, comparisons, uncertainty, timings and plots |
| `paper/main.tex`, `paper/main.pdf` | Complete twelve-page Evaluation 2 manuscript |
| `evaluation2_paper_overleaf.zip` | Self-contained paper source and figure package |
| `README.md`, `DATA_ACQUISITION.md` | Setup and reproduction commands |

## What is published and what stays local

The repository contains the source, tests, documentation, compact report outputs, figures, compiled paper and Overleaf package. Raw WESAD recordings, generated model bundles, large row-level prediction CSVs, virtual environments, caches and temporary LaTeX files remain excluded by `.gitignore`. The original tracked feature CSV and baseline model are preserved. Follow `README.md` to regenerate the new data and trained artifacts after cloning; the paper can be compiled independently from its included assets.

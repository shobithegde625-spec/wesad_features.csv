# Paper-ready results: adaptive multimodal stress monitoring

These are completed experimental results, not projected improvements. The evaluation holds out each of 15 WESAD participants in turn, scores 434 fully labeled 60-second replay windows, and includes 11 clean/corruption scenarios. Models, probability calibration and routing thresholds use separate development participants within each outer fold. This revision is exploratory because the cohort has already informed development.

## What was implemented, in simple words

1. Start with ECG, which measures the heart's electrical activity.
2. Check whether the signal is usable and whether the ECG model has enough confidence to answer.
3. When more evidence is needed, combine ECG with skin conductance (EDA) and breathing. If ECG fails, use the independent EDA-plus-breathing model.
4. Return insufficient evidence when usable measurements are unavailable. Do not turn missing signals into a baseline prediction.

The revision adds grouped SVM parameter selection and a compact forest, calibrates confidence on different participants from those used for fitting, separates the threshold for requesting more sensors from the threshold for accepting a final answer, and adds a high-frequency ECG noise check. The software also includes chronological replay, a dashboard and versioned experiment records. Forty-one automated tests passed, followed by a successful dashboard check against the final v2 artifact.

## Three observations to showcase

### 1. Multimodal adaptive processing detects more stress than the original-feature ECG baseline

On the same clean held-out windows:

| Method | Accuracy | All-window stress recall | Stress F1 | Coverage |
| --- | ---: | ---: | ---: | ---: |
| Original seven-feature ECG SVM baseline | 83.18% | 74.34% | 0.756 | 100% |
| Proposed adaptive cascade | 85.48% | 88.16% | 0.810 | 100% |

Observation: stress recall rises by **13.82 percentage points**, accuracy by **2.30 points**, and F1 by **0.054**. In simple words, the proposed system catches more stress windows than the original-feature ECG baseline. This compares complete pipelines; the gain cannot be attributed to routing alone because features, expert training and calibration also differ. Stress precision is slightly lower (74.86% versus 76.87%), so improved recall comes with a false-positive tradeoff. These are descriptive cohort results, not a statistically established superiority claim.

### 2. The held-out cascade approaches fusion's stress recall with less processing

| Method | All-window stress recall | Mean processing CPU, ms/window |
| --- | ---: | ---: |
| Always-on fusion | 88.82% | 16.77 |
| Proposed adaptive cascade | 88.16% | 14.24 |

Observation: the cascade uses **15.08% less CPU** while stress recall is only **0.66 percentage points lower**. It requests fusion for 37.56% of clean scored windows. In simple words, consulting extra sensors only when needed can reduce average computation while retaining most stress detections in the held-out procedure.

Scope: the efficiency result belongs to the outer-fold training-procedure evaluation. The separately fitted dashboard artifact did not reproduce CPU saving on its representative replay benchmark (19.83 ms versus fusion's 19.65 ms). The original 25% CPU-saving target remains unmet. Overall accuracy also remains below fusion (85.48% versus 88.71%). Avoid the broader claim that accuracy is unchanged or that the deployed artifact saves 15% CPU.

### 3. Independent alternate sensors preserve useful operation when ECG fails

| Failure scenario | Proposed accuracy | Stress recall | Stress F1 | Coverage |
| --- | ---: | ---: | ---: | ---: |
| ECG flatline | 85.25% | 77.63% | 0.787 | 100% |
| Severe synthetic ECG noise | 85.25% | 77.63% | 0.787 | 100% |
| Total sensor dropout | Undefined | 0% | Undefined | 0% |

Under ECG flatline, the ECG-only baseline and full-modality-required fusion baseline cannot produce usable decisions. The cascade continues through its independent EDA/respiration expert. Under severe noise, the new quality check rejects ECG and uses the same alternate path; identical outcomes in these two scenarios are therefore expected.

Relative to the previous cascade, severe-noise F1 increases from **0.510 to 0.787**, and precision from **35.06% to 79.73%**. Stress recall decreases from 93.42% to 77.63%; the improvement is fewer false stress predictions, not improved recall under that noise condition. In simple words, the system can still use the remaining sensors instead of confidently trusting a damaged ECG signal. With every sensor unavailable, it explicitly declines to classify.

## Suggested results paragraph

In an exploratory fifteen-subject leave-one-subject-out WESAD evaluation, the proposed quality-aware adaptive cascade achieved 85.48% accuracy, 88.16% all-window stress recall and 0.810 stress F1 at full coverage on 434 eligible replay windows. Relative to the original-feature ECG SVM baseline, stress recall improved by 13.82 percentage points. Compared with always-on fusion, mean processing CPU time decreased by 15.08%, with a 0.66-percentage-point reduction in stress recall. Under ECG flatline and severe synthetic ECG noise, alternate EDA/respiration inference retained 85.25% accuracy at full coverage. These findings support a useful laboratory accuracy–computation tradeoff and ECG-failure resilience, while independent validation and stable fitted-artifact efficiency remain necessary.

## Limitations to retain in the paper

- The independent sample is 15 people. Participant intervals are broad and do not establish formal superiority or noninferiority. Calibration and policy selection are unstable across participant splits.
- The final fitted artifact has no demonstrated CPU saving on its benchmark; cross-validation savings cannot be assigned to that artifact. CPU time measures computation, not battery energy or physical edge-device performance.
- Relative to v1, clean pooled F1 declines from 0.831 to 0.810. Tripling ECG amplitude reduces v2 stress recall to 53.95%; alternate-sensor dropout also reduces recall despite full coverage. Robustness is scenario-specific.
- Event-onset detection remains weak. Window classification results should not be presented as reliable stress-episode alerts. Each decision uses a completed 60-second window.

## Evidence and figures

- [Full revised evaluation](reports/multimodal-real-v2/REPORT.md)
- [Verified version comparison](reports/multimodal-revision-comparison/FINDINGS.md)
- [Exportable figure, PDF](reports/multimodal-revision-comparison/revision_tradeoff.pdf) and [PNG](reports/multimodal-revision-comparison/revision_tradeoff.png)
- [Detailed claim, uncertainty and artifact benchmark](RESEARCH_CLAIM.md)
- [Per-subject metrics](reports/multimodal-real-v2/per_subject.csv)

Final dashboard artifact: `artifacts/multimodal-real-v2/model.joblib`, ID `dae3a7c563c9`. The original data, artifacts and reports remain available. No additional model fitting or test-set tuning was performed to produce this summary.


The complete Evaluation 2 manuscript, following FISAC QP and retaining the earlier section flow, is available in [paper/main.tex](paper/main.tex). See [paper/README.md](paper/README.md) for figures, compilation and submission notes.

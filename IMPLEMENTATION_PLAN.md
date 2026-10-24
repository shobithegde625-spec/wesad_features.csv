# Implementation plan: quality-aware adaptive stress monitoring at the edge

Recommended extension: build a lightweight cascade that uses ECG for straightforward windows, consults EDA and respiration when more evidence is needed, and abstains when signal quality or confidence is insufficient. Convert accepted predictions into sustained stress events rather than alerting independently on every window.

The research question is: **Can quality-aware escalation reduce false stress alerts and feature-processing cost compared with always using all modalities, while maintaining stress-event recall on unseen subjects?**

This plan follows a review of all 13 Python files, requirements.txt, the feature CSV, saved comparison results, and the seven-page EdgeBased_Healthcare.pdf. It is a plan only; no models were retrained or benchmarked during the review.

**What the project currently establishes**

| Area | Evidence in this repository | Implication |
| --- | --- | --- |
| Data | 445 rows, 15 subjects, 285 baseline and 160 stress windows | Small number of independent participants; evaluate uncertainty across people. |
| Extraction | prepare_dataset.py uses chest ECG at 700 Hz, nonoverlapping 60-second windows, seven amplitude statistics | No multimodal fusion or beat-based physiological features yet. |
| Labels | Original labels 1 and 2 become baseline 0 and stress 1; windows stay inside continuous label segments | Good alignment for classification, but transitions and other conditions are discarded. |
| Evaluation | GroupShuffleSplit in training; five-fold GroupKFold in validation/comparison | Subject separation is already present and should be preserved throughout tuning and calibration. |
| Saved comparison | SVM: balanced accuracy 0.806, stress F1 0.752; RF: 0.782 and 0.721; logistic regression: 0.741 and 0.672 | These are saved results, not newly reproduced measurements. |
| Deployment | train_model.py saves a 100-tree RF; compare_models.py evaluates a 200-tree RF in a pipeline | Comparison metrics do not identify the exact deployed estimator. |
| Probabilities | RF predict_proba is displayed without calibration; comparison SVC has probability=False | Confidence-based routing needs an explicit calibration implementation. |
| Monitoring | app.py replays CSV rows using a UI timer and wraps around at the end | Playback time is not physiological time; state currently spans subjects and replay boundaries. |
| Edge evidence | Local inference and model-size scripts exist; saved bundle is 514,937 bytes, about 503 KiB | No target-device energy or end-to-end feature-processing measurement yet. |
| Reproducibility | Raw WESAD path is hardcoded; NumPy, SciPy and Matplotlib are imported but not directly declared | Dataset regeneration and environment setup need explicit configuration. |

The configured raw-data directory, ~/OneDrive/Documents/WESAD, is absent in this environment. The current CSV contains no raw waveforms, sample positions, segment IDs, or quality annotations. New physiological features and meaningful event evaluation require regenerating data from raw WESAD; they cannot be recovered from these seven statistics.

**Why this extension is worthwhile, and what “novel” means here**

The PDF's research gap and methodology, on pages 4–5, call for multimodal processing with measured edge efficiency. Page 6 already proposes personalization, fusion and model compression. Simply adding one of those is an implementation extension, but a weak standalone research contribution.

Adaptive multimodal sensing and cost-aware feature selection also have prior art, including [CAFS](https://lup.lub.lu.se/search/publication/ae40aea8-3a17-4ed9-b975-c5029ce8e999) and [AMSER](https://users.utu.fi/spakan/wp-content/uploads/sites/57/2022/03/amser_date22.pdf). Abstention is established through [selective classification](https://proceedings.neurips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html).

The proposed contribution is a specific, reproducible policy and evaluation: **quality-aware routing plus abstention plus temporal event decisions, assessed jointly for corruption robustness, class-specific coverage, alert quality and measured edge cost.** This is a defensible project-level contribution and a research hypothesis. Publication-level originality requires a deeper comparison with existing policies and evidence that this implementation improves their tradeoff.

**Step 1 — Freeze and reproduce the existing baseline**

- Keep the existing CSV and model intact. Put new outputs under versioned artifacts/ and reports/ directories.
- Add command-line configuration for data paths, output paths and random seed. Move script execution behind main guards so modules can be imported safely.
- Declare direct dependencies, then record exact resolved versions for reproducible runs. Inspect the existing bundle in a compatible environment before assuming it loads under newer scikit-learn versions.
- Centralize feature order, label mapping, model factories and dataset validation. Preserve the seven original features as an explicit baseline.
- Make train_model.py, validate_model.py and compare_models.py use the same estimator definitions. Store estimator parameters, subject assignments, dataset hash, versions and metrics with each run.
- Reproduce the saved comparison before extending it; explain any differences rather than silently replacing results.

Completion evidence: one command regenerates a baseline report with exact train/test subjects and an artifact ID that the dashboard can display. Malformed input or an absent raw directory fails clearly without overwriting artifacts.

**Step 2 — Establish an evaluation protocol before tuning**

- Use leave-one-subject-out evaluation as the primary outer protocol: 15 outer folds. Retain the original five-fold grouped result as a historical comparison.
- Within each outer training set, use subject-disjoint development partitions for model selection, probability calibration and policy tuning. Make these roles explicit in a saved split manifest; use grouped cross-fitting where necessary to use the limited data efficiently.
- All imputation, scaling, feature selection, quality thresholds, calibrators and routing/alert thresholds must be fitted using development subjects only. The outer subject remains untouched until final evaluation.
- For any calibration implementation, pass explicit subject-disjoint index splits. The default integer CV in CalibratedClassifierCV uses stratification rather than participant separation; see the [official API documentation](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html).
- Save row-level held-out predictions with subject and artifact IDs. Report participant-macro metrics alongside pooled metrics, per-subject results, and subject-bootstrap intervals. Do not bootstrap windows as independent people.

Completion evidence: automated checks prove that outer test subjects never enter fitting or tuning, and all compared methods receive identical outer test windows.

**Step 3 — Regenerate a timestamped multimodal dataset**

- Extend prepare_dataset.py to accept a raw WESAD root and write a separate dataset version. Begin with chest ECG, EDA, respiration and acceleration to simplify initial synchronization.
- Verify sampling rates, lengths, units and timestamps from the dataset documentation. The [original WESAD paper](https://kristofvl.github.io/usi/pdf/ubi_icmi2018.pdf) describes the available chest and wrist modalities; do not assume wrist streams have the chest sampling rate.
- Preserve subject, session, continuous segment ID, start/end sample, start/end time, original condition, binary target label when applicable, window length, stride and source-file identity.
- Reproduce the current 60-second nonoverlapping pure-condition windows first. Create a separate chronological replay dataset with explicit gaps, transitions and unsupported conditions; never join disconnected periods into a synthetic continuous recording.
- Keep labels outside the inference API. Labels may select training/evaluation targets, but must never determine which expert or routing branch runs.
- Add 30-second windows only as a planned secondary experiment after the 60-second pipeline works. If using overlapping windows, retain subject grouping and account for their temporal dependence.

Completion evidence: feature rows map back to exact waveform intervals; extraction does not cross subjects or silently span gaps; classification and replay datasets have documented inclusion rules.

**Step 4 — Add physiological features and independent quality checks**

- Retain the seven ECG amplitude statistics for comparison. Add heart rate, valid beat count, median RR, SDNN and RMSSD from an explicit R-peak detection pipeline. Describe them as short-window estimates; defer frequency-domain HRV until window-length assumptions are justified.
- Add a compact EDA feature set: level summaries, slope and a documented phasic-response measure. Add respiration rate and variability only when the signal supports reliable cycle detection.
- Use acceleration as movement context and a quality aid. Movement alone must not imply stress or prove signal corruption.
- Compute low-cost quality indicators before expensive features: missing fraction, flatline fraction, clipping, beat/cycle validity and artifact indicators. Learn any data-dependent bounds inside development folds.
- Represent unavailable modality features and extraction failures explicitly. A missing ECG should route to a supported alternate expert or abstention, never to an invented normal measurement.
- Validate signal-processing behavior on known-rate synthetic signals and manually inspect representative waveform/peak overlays. Share identical feature functions between offline extraction and runtime inference.

Completion evidence: flatlined or missing signals produce documented quality failures; feature outputs are finite when valid; training and replay extraction agree for the same interval.

**Step 5 — Build calibrated experts and fair static baselines**

- Train a compact ECG expert, an EDA-plus-respiration expert for ECG failure, and an ECG-plus-EDA-plus-respiration fusion expert.
- Start with logistic regression, SVM and bounded-size Random Forests. Use the original seven-feature ECG model as the first baseline, then the same ECG model family with physiological features.
- Include an always-on fusion baseline using the same feature extractors and available experts as the cascade. Otherwise improvements could come from different models rather than routing.
- Begin with sigmoid calibration because the dataset is small. Use subject-disjoint fitting and calibration data; evaluate reliability diagrams, Brier score and log loss. Calibration is empirical and does not guarantee reliability on a new person or a corrupted signal.
- If a compact linear model approaches the stronger model's performance, retain it as a candidate based on measured runtime and size.

Completion evidence: each expert has untouched outer-fold probabilities, quality metadata, feature schema and model size. The final chosen pipeline, rather than a differently configured estimator, is what gets packaged.

**Step 6 — Implement the quality-aware routing policy**

Implement the following deterministic initial policy in a new routing.py module:

```text
Inspect low-cost signal quality and modality availability.
If ECG is usable:
    Compute ECG features and calibrated ECG probability.
    If probability passes the development-selected confidence rule:
        Return an accepted ECG decision.
    Otherwise, if EDA and respiration are usable:
        Compute their features and run the fusion expert.
        Accept only if its quality and confidence rules pass.
If ECG is unusable and EDA and respiration are usable:
    Run the alternate EDA/respiration expert and its acceptance rule.
Otherwise, or if the selected expert remains uncertain:
    Return insufficient_evidence with a reason.
```

- Handle missing alternate channels explicitly. The first version may abstain when only one alternate channel remains; additional experts are a later experiment.
- Tune class-specific confidence thresholds on development subjects, with an explicit coverage floor and stress-recall constraint. Avoid minimizing errors by rejecting nearly all stress windows.
- Assess expert calibration on the routed subset as well as globally: the fusion model sees a harder subset during cascade operation. Any routing-specific recalibration also uses development data only.
- Return probability when available, accepted/rejected status, class when accepted, selected expert, quality flags, rejection reason, feature time, inference time and artifact version.
- Cache shared features so escalation does not recompute ECG work. Never read label or subject-specific outcome information when routing.

This initially saves feature-processing and inference work on already available streams. It does not establish sensor acquisition or radio savings. Quality checks themselves consume work and must be included in runtime measurements. Confident ECG mistakes remain a known weakness; quantify them directly in the stress tests.

Completion evidence: branch-level tests cover confident ECG, uncertain ECG, ECG failure with alternate success, alternate failure, and total dropout. Routing outputs remain identical when evaluation labels are removed.

**Step 7 — Convert windows into sustained events**

- Add temporal_alerts.py with separate state per subject/session and reset it at gaps, subject changes and replay restarts.
- Require sustained accepted stress evidence to open an event; use a lower recovery threshold or sustained baseline evidence to close it. Tune confirmation duration and cooldown on development subjects.
- Treat rejected windows as unknown. Do not count them as baseline, carry accepted evidence across long unknown gaps, or allow an old alert to imply current confidence indefinitely.
- Keep raw probability, accepted window decision and event state distinct in results.
- Evaluate using source timestamps. A 60-second window followed by a confirmation interval creates detection delay even if model inference takes milliseconds.
- Initially use completed, trailing windows and causal processing. Do not use centered filters or future samples in an online-latency claim.

Completion evidence: isolated probability spikes do not create repeated alerts, sustained accepted evidence does, missing data transitions to unknown, and a participant boundary clears all state.

**Step 8 — Test the contribution with ablations and sensor failures**

| Experiment | Purpose |
| --- | --- |
| Original seven-feature ECG | Reproduce the starting point. |
| Physiological ECG features | Isolate feature engineering. |
| Always-on multimodal fusion | Establish what full computation achieves. |
| Confidence-only cascade | Test adaptive routing without signal-quality gates. |
| Full quality-aware cascade | Isolate the quality-aware routing contribution. |
| Full cascade without abstention | Quantify the value and coverage cost of rejection. |
| Full cascade with/without temporal confirmation | Isolate event-policy effects. |
| Matched-budget and matched-coverage comparisons | Prevent unfair cost or rejection advantages. |

- Start with clean held-out recordings. Then apply reproducible missing bursts, flatlines, clipping, amplitude scaling and noise at predeclared severities to held-out raw signals. Rerun quality checks and feature extraction; corrupting seven summary numbers is not an ECG artifact experiment.
- Apply single-channel and combined channel failures. Record corrupted windows and original labels separately. Do not select corruption severity by inspecting test performance.
- Include development-only corruption when tuning robustness; reserve additional failure combinations or severities for final stress tests.
- Report balanced accuracy, stress precision/recall/F1, Brier score, accepted coverage, class-specific coverage, selective error, and risk-versus-coverage curves. Never report accepted-only accuracy without rejection rates.
- Count abstained stress windows as undetected for an all-window operational recall measure, while reporting accepted-only classification metrics separately.
- For chronological replay, report false event starts per eligible baseline hour, stress-episode detection rate and onset delay. Unsupported conditions and unknown gaps are excluded or reported separately, never silently called baseline.
- WESAD has few laboratory stress episodes per person. Event metrics support a laboratory demonstration, not a claim about day-long real-world false-alert rates. Transition-inclusive binary classification needs a predefined evaluation rule; unsupported transition windows remain unscored for binary accuracy.

Completion evidence: one report compares all variants on identical outer subjects and includes failure cases, per-subject variation, uncertainty and tradeoff plots.

**Step 9 — Measure the full edge processing path**

- Refactor edge_inference.py into a reusable runtime that accepts waveform windows or a replay source, not a hand-picked labeled CSV example.
- Extend benchmark_inference.py to measure quality checking, feature extraction, expert inference, routing, event logic and total processing time. Report cold startup separately from steady-state p50/p95/p99 latency.
- Include representative clean and corrupted windows and each routing branch, not 100 repetitions of one stored feature row. Disable oversubscribed threading when benchmarking single-device behavior.
- Measure combined bundle size, loaded process memory, escalation fraction and CPU time on the same target device as always-on fusion. Choose an accessible Linux edge computer, such as a Raspberry Pi, before setting deployment claims; desktop results remain desktop results.
- Report acquisition/window duration separately from processing latency. Record hardware, OS, library versions, thread settings and benchmark sample count.
- Measure energy with appropriate power instrumentation only if available. Without it, report compute/time proxies explicitly and make no battery-life claim.

Completion evidence: the cascade's aggregate cost includes all experts and gates, and results can show whether their overhead outweighs avoided processing.

**Step 10 — Integrate the runtime into the existing dashboard**

- Keep Streamlit as the interface; put inference, routing and event state transitions in ordinary Python modules.
- Add subject/session selection and chronological source time to Monitoring. Stop at replay completion by default; any loop starts a new session and resets state.
- Show accepted baseline, accepted stress and insufficient evidence distinctly. Present event state, quality reasons, selected expert and processing latency beside the source-time trend.
- Extend History exports with source interval, quality flags, route, abstention reason, window decision, event state and artifact ID.
- Tie Performance metrics to the loaded artifact and evaluation protocol. Historical results may remain visible if clearly labeled with their own artifact identity.
- Keep the first version fully local and prerecorded. A live adapter can implement the same runtime input contract after replay is validated.

Completion evidence: a demo can replay a clean session and a corrupted copy, visibly explain alternate routing and rejection, and export results consistent with the evaluation report.

**Step 11 — Apply explicit success criteria and decide what to keep**

Set these as provisional engineering targets before final outer evaluation, then freeze the operating-point selection rule using development data:

- Relative to always-on fusion: at least 25% lower mean total processing CPU time, with all-window stress recall no more than 3 percentage points lower.
- Accepted coverage at least 80% overall on clean windows; always disclose stress and baseline coverage separately and predefine a stress coverage floor during development.
- Relative to a simple single-window threshold alert: at least 20% fewer false event starts, while reporting event recall and additional onset delay. If the baseline has zero false starts, report absolute differences rather than a percentage.
- On corruption/dropout tests: lower error among accepted decisions at matched coverage, with all-window stress detection and rejected time disclosed.
- Target-device p95 processing time below one second per completed 60-second window as an initial responsiveness budget. This excludes window acquisition and temporal confirmation time.

These are targets, not predicted outcomes or statistical guarantees. With only 15 participants, show paired participant differences and intervals before making superiority claims. If the cascade does not outperform static fusion on the joint cost/reliability tradeoff, keep the useful quality checks and abstention and report the negative result honestly.

**Suggested delivery order**

1. Milestone A, roughly 3–5 working days: reproducible baseline, shared schemas, grouped evaluation and artifact identity. This can start immediately with the existing CSV.
2. Milestone B, roughly 1 week after raw-data access: timestamped multimodal extraction, quality checks and static experts.
3. Milestone C, roughly 1 week: calibrated routing, abstention, temporal decisions and focused correctness tests.
4. Milestone D, roughly 1 week: outer evaluation, corruption experiments, ablations and target-device benchmarks.
5. Milestone E, roughly 2–3 working days: dashboard integration and a reproducible final report.

Estimate: about 4–5 weeks for one developer comfortable with Python and signal processing, assuming raw data and hardware are available. Hardware integration or investigating unreliable beat detection may extend this.

The first implementation task should be the shared baseline/evaluation infrastructure, not dashboard changes. It establishes the evidence needed to judge whether the extension improves the project.

**Scope to defer**

Defer federated learning, transformers, a mobile app, automatic online retraining, cross-device sensor scheduling and model compression until this experiment has a result. Personalization is a separate follow-up: it requires an explicit onboarding protocol and exclusion of any calibration interval from scored test windows. A lightweight CNN can be a later comparison if completing every candidate in the PDF is a course requirement; it is not necessary to test this contribution.

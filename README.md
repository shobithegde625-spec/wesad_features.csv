# Pulsepoint

A local stress-monitoring research prototype with reproducible participant-held-out evaluation, calibrated ECG/multimodal experts, quality-aware routing, abstention and causal event decisions.

The original CSV, Random Forest bundle, comparison CSV and figures are preserved. New training writes separate versioned artifacts; new evaluation writes separate reports. No sensor data is uploaded by this implementation.

The official archive is downloaded and verified at `data/downloads/WESAD.zip`. Revised multimodal training, all fifteen held-out folds across eleven scenarios and 41 automated tests have completed. The dashboard defaults to `artifacts/multimodal-real-v2/model.joblib` and fills its replay/source paths automatically. Version 1 artifacts and results remain available.

Read [the supported research claim and limitations](RESEARCH_CLAIM.md), [the before/after findings](reports/multimodal-revision-comparison/FINDINGS.md), [the acquisition record](DATA_ACQUISITION.md), and [the implementation status](IMPLEMENTATION_STATUS.md). The revised cascade reaches 88.2% clean all-window stress recall at full coverage, versus fusion's 88.8%, using 15.1% less CPU in the same evaluation. The 25% saving target remains unmet. ECG failure/noise handling improves; pooled clean F1, amplitude-scaling robustness and alternate-dropout recall show tradeoffs or regressions. Episode-onset alerting remains experimental.

The separate final-artifact benchmark is cost-neutral: approximately 19.83 ms mean CPU for cascade versus 19.65 ms for fusion on the sampled clean replay workload. The held-out training procedure's saving cannot be claimed for this fitted dashboard artifact.

## Run the existing-data implementation

Use Python 3.10 or newer. This workspace was validated with Python 3.14.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python train_model.py
.venv/bin/python -m streamlit run app.py
```

A trained ECG-summary bundle is already available locally at `artifacts/ecg-summary-v1/model.joblib` (a newer final bundle may also be present). The dashboard selects the newest complete artifact automatically. Generated model files are ignored by Git; train again after cloning. The original `stress_detection_model.joblib` is retained for historical reproducibility and is not silently loaded as a new calibrated model.

```bash
# Primary nested LOSO evaluation: all 15 participants are held out in turn.
.venv/bin/python evaluate.py --output reports/my-loso-run

# Reproduce the original five-fold result, including its 200-tree forest.
.venv/bin/python compare_models.py --families logistic svm historical_forest --output reports/my-historical-run

# Replay stored summaries. Substitute your own artifact path after retraining.
.venv/bin/python edge_inference.py --bundle artifacts/ecg-summary-v1/model.joblib --subject S2 --limit 10

# Host benchmark: this excludes feature extraction for the original CSV.
.venv/bin/python benchmark_inference.py --bundle artifacts/ecg-summary-v1/model.joblib --output reports/my-benchmark

.venv/bin/python -m pytest
```

Output directories must be new (evaluation also accepts an empty directory). Existing model/data files are never overwritten. Every training/evaluation run records the seed, resolved package versions, feature order, model parameters, dataset and implementation hashes, participant roles and artifact identities. Exact installed versions from this implementation session are in `requirements-lock.txt`; that lock includes platform-specific dependencies and Python 3.14 wheels, so use `requirements-dev.txt` on another platform if needed.

## Enable the full multimodal implementation

The raw-data adapter accepts either an extracted directory or the official ZIP. Direct ZIP access avoids expanding approximately 12.9 GiB of synchronized recordings. It loads one participant at a time, retains only the pipeline channels, verifies entry CRCs, and converts only the requested window to floating point.

To download and verify the original archive:

```bash
.venv/bin/python download_wesad.py --output data/downloads/WESAD.zip
.venv/bin/python import_wesad.py --archive data/downloads/WESAD.zip --archive-only --output data/wesad-documentation
.venv/bin/python prepare_dataset.py --raw-root data/downloads/WESAD.zip --output data/wesad-real-v1
```

The downloader uses at most four range requests and verifies the official host's SHA-1 before publishing the archive. It can reuse a sequential partial file with `--prefix-file path/to/WESAD.zip.part`. The importer writes SHA-256 provenance, archive-member identities, documentation and usage terms. The data permits scientific, non-commercial research with attribution; cite Schmidt et al. (2018), DOI 10.1145/3242969.3242985, when publishing results.

Use the ZIP path in place of `/path/to/WESAD` in the remaining commands. The dashboard automatically fills the replay CSV and raw source from a multimodal dataset's metadata.

Obtain the original [WESAD recordings](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html) and unpack them so that the raw root contains `S2/S2.pkl`, `S3/S3.pkl`, etc. WESAD pickles and model bundles should be trusted local files. The original seven-summary CSV cannot reconstruct these waveforms.

```bash
.venv/bin/python prepare_dataset.py --raw-root /path/to/WESAD --output data/wesad-v1

.venv/bin/python train_model.py --data data/wesad-v1/classification.csv --raw-root /path/to/WESAD --output artifacts/multimodal-v1

.venv/bin/python evaluate.py --data data/wesad-v1/classification.csv --replay data/wesad-v1/replay.csv --raw-root /path/to/WESAD --scenarios clean ecg_missing_burst ecg_flatline ecg_clipping ecg_noise_moderate ecg_noise_severe ecg_amplitude_scale eda_dropout resp_dropout alternate_dropout total_dropout --output reports/multimodal-v1

.venv/bin/python benchmark_inference.py --bundle artifacts/multimodal-v1/model.joblib --data data/wesad-v1/replay.csv --raw-root /path/to/WESAD --scenarios clean ecg_flatline total_dropout --output reports/multimodal-benchmark

.venv/bin/python feature_diagnostics.py --raw-root /path/to/WESAD --subject S2 --start-seconds 300 --output ecg-overlay.png
```

Run the benchmark on the actual edge device to assess device latency. CPU time is a processing proxy; energy/battery claims require power instrumentation. Window acquisition, disk IO and communications are excluded from the processing timer and documented separately.

For a 30-second secondary experiment, prepare a separate directory with `--window-seconds 30`. This first implementation deliberately requires nonoverlapping windows. Chest streams use the documented 700 Hz alignment; wrist fusion requires a separate rate-aware adapter.

Select the multimodal bundle and `replay.csv` in the dashboard, enter the raw WESAD directory, and choose a participant/recording. Corruption scenarios apply to waveforms before feature extraction. Without a raw directory, the dashboard uses stored features and explicitly excludes extraction time. Replay ends without wrapping; restarts, participant changes, gaps and unknown observations clear temporal evidence.

## How the implementation works

New training uses the revised method documented in [MODEL_IMPROVEMENT.md](MODEL_IMPROVEMENT.md): participant-weighted grouped hyperparameter selection, independent ECG exit/final acceptance thresholds, fusion-relative development constraints and an optional high-frequency ECG quality gate. Existing v1 bundles keep their original routing and quality behavior. When a revised bundle requests fusion but its alternate channels are unavailable, a usable ECG can still pass its final acceptance rule, with that reason included in the result.

Compare complete versions on verified identical windows and participant splits:

```bash
.venv/bin/python compare_experiments.py --previous reports/multimodal-real-v1 --current reports/multimodal-real-v2 --output reports/multimodal-revision-comparison
```

The comparison writes participant-paired bootstrap differences and checks unchanged success criteria. Revised WESAD results remain exploratory because this cohort has already informed development; nested grouping alone does not turn the revised experiment into independent confirmation.

- `pulsepoint/prepare.py` creates pure-condition classification windows and a separate chronological replay table. Both preserve source sample/time intervals and source-file identities. Mixed or unsupported replay conditions have no binary target and are excluded from binary scoring.
- `pulsepoint/features.py` extracts original amplitude summaries, short-window heart-rate/RR estimates, EDA level/slope/response proxies and respiration features. Fixed quality heuristics detect missing data, flatlines, repeated extrema and unreliable beat/cycle extraction. Acceleration is exported as movement context; movement is not a stress label. These are research heuristics requiring real-signal validation.
- `pulsepoint/models.py` selects logistic regression, SVM or Random Forest within fitting participants using grouped CV. A separate participant set fits a sigmoid calibrator. A third set selects the routing policy. No participant from an outer test fold enters those roles. There is no implicit window-stratified calibration.
- `pulsepoint/routing.py` accepts confident ECG decisions, escalates ambiguous ECG to fusion, uses the EDA/respiration expert for ECG failure, and abstains when evidence is insufficient. Features are cached once per waveform window. Runtime routing consumes declared features and quality, never labels or participant identity.
- `pulsepoint/training.py` can add development-only ECG missing/noise and EDA-dropout scenarios when raw data is supplied. Severe noise, clipping and combined dropouts remain additional stress tests. Coverage and recall constraints are checked on clean and combined development data. Infeasible constraints are disclosed instead of claimed as satisfied.
- `pulsepoint/temporal.py` separates accepted window decisions from sustained events. Opening, recovery and cooldown durations are selected on policy participants when timestamps exist. Rejections are unknown, never baseline. The original CSV has no source timestamps, so event inference is disabled for it.
- `pulsepoint/replay.py` is shared by the dashboard, CLI, evaluation and benchmark. All signal processing uses only the completed trailing window; nothing requires future windows.

Sigmoid calibration uses disjoint participant sets, consistent with the [scikit-learn calibration guidance](https://scikit-learn.org/stable/modules/calibration.html). Calibration does not ensure reliability on a new person, a routed subset or a corrupted signal; reports include reliability diagrams globally and per selected expert.

## Read the evidence

Each evaluation directory contains:

- `REPORT.md`, `summary.json`, `comparison.csv` and `evaluation.png`.
- `predictions.csv`, `split_manifest.json` and per-fold persisted models under `models/`.
- `per_subject.csv`, participant-bootstrap intervals and `paired_differences.json`.
- Risk–coverage curves, reliability bins and reliability bins for routed subsets.
- `matched_coverage.csv`: posthoc confidence-ranking diagnostics on accepted predictions, explicitly not a deployed policy or test-time threshold selection.
- `budget_frontier.csv`: measured fixed-policy candidates at different processing budgets. No winning policy is selected from test outcomes.
- `targets.json`: predeclared engineering checks; unavailable multimodal/hardware evidence remains pending.

Accuracy and F1 refer to accepted decisions. All-window stress recall counts rejected stress as undetected. Coverage is reported overall and separately by class. Probability metrics disclose the fraction of windows with available probabilities. Event statistics use source time and eligible baseline hours, not the dashboard playback timer. Episodes and onset delay are defined from fully labeled replay windows; mixed transition windows are unscored, so delay is relative to the first eligible stress-window start rather than the exact protocol transition. Confidence-ranked diagnostic curves do not establish statistical guarantees.

The existing-data run reproduces the original historical model comparison exactly. The new CSV-only nested cascade does **not** outperform the original SVM on stress recall. This is a negative result for the summary-only calibrated cascade, not evidence for multimodal benefit. Inspect `reports/loso-final/REPORT.md` for the final run and `reports/historical-reproduction/REPORT.md` for reproduction evidence.

The official raw archive is now downloaded and checksum-verified; see DATA_ACQUISITION.md and reports/raw-data-validation.json. Real multimodal training is saved under artifacts/multimodal-real-v1/, and the full held-out corruption experiment writes reports/multimodal-real-v1/. The original summary-only negative result remains available separately. Synthetic tests verify software behavior and remain distinct from real-data performance evidence.

## Remaining evidence work

The implementation covers the planned software changes, and the official dataset is available locally. A physical edge-device benchmark, real-world stress specificity, long-duration false-alert rates and energy savings still require additional recordings/hardware beyond this repository. Benchmark process RSS includes the offline participant cache and allocator retention, so it is not the deployed model working set.

Federated learning, transformers, personalization and a mobile app remain deferred as specified in the implementation plan. With 15 laboratory participants, report participant variation and uncertainty rather than claiming broad real-world generalization.

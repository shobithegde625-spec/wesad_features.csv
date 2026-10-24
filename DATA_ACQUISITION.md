# Verified WESAD acquisition

The original university-hosted WESAD archive is available locally at `data/downloads/WESAD.zip`. It was retrieved from the download linked by the [official dataset page](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html).

- Archive size: 2,249,444,501 bytes.
- Official host's SHA-1, verified against the downloaded bytes: `bae786bf599ef28e42e1c2e18da9b84d685e84e7`.
- Locally recorded SHA-256: `5e15d2606adf16d819ba535c1786d92e94bae5ff5392f7ae01fb63f9938fd71c`.
- Synchronized recordings: 15 participants, S2–S17 excluding S12 (S1 is also absent from the released data).
- Terms: scientific, non-commercial use with attribution to the dataset owners. Consult the official page before other uses.

The synchronized pickle entries expand to approximately 12.9 GiB, exceeding this computer's available disk space. The pipeline therefore reads directly from the verified ZIP. It caches only one participant's required channels and checks each ZIP entry's CRC while loading. The archive contains both chest and wrist data; this implementation uses the synchronized 700 Hz chest ECG, EDA, respiration and acceleration streams.

The official PDF and all participant recording notes are in `data/wesad-documentation/`. Its `acquisition.json` records the source URL, checksums, file identities and usage terms. Download chunks were removed after successful assembly and verification.

Real-data preparation produced:

| Table | Windows | Purpose |
| --- | ---: | --- |
| `data/wesad-real-v1/classification.csv` | 445 | Pure-condition training/evaluation intervals: 285 baseline, 160 stress. |
| `data/wesad-real-v1/replay.csv` | 1,442 | Chronological, nonoverlapping replay of complete recordings. |
| Binary-scored replay subset | 434 | 282 baseline and 152 stress windows. |
| Unsupported/mixed replay intervals | 1,008 | Processed chronologically but excluded from binary scoring. |

The regenerated seven ECG amplitude statistics, participant order and labels reproduce the original project CSV within a numerical tolerance of 1e-10. All 445 classification windows pass the current modality-quality heuristics. Two respiration intervals fail quality during full replay. These heuristic checks are not clinical signal-quality annotations.

See `reports/raw-data-validation.json` for machine-readable verification. A qualitative waveform/energy-peak diagnostic for a baseline interval from fitting participant S9 is saved at `reports/multimodal-diagnostics/S9-baseline-waveforms.png`; it does not establish detector accuracy across the dataset.

Reproduce acquisition and preparation:

```bash
.venv/bin/python download_wesad.py --output data/downloads/WESAD.zip
.venv/bin/python import_wesad.py --archive data/downloads/WESAD.zip --archive-only --output data/wesad-documentation
.venv/bin/python prepare_dataset.py --raw-root data/downloads/WESAD.zip --output data/wesad-real-v1
```

Existing output paths are preserved; use new paths when repeating a completed run. The ZIP path can replace the raw directory in training, inference, inspection, diagnostics, benchmarking and evaluation commands.

Reference: Philip Schmidt, Attila Reiss, Robert Duerichen, Claus Marberger and Kristof Van Laerhoven (2018), *Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection*, ICMI. DOI: [10.1145/3242969.3242985](https://doi.org/10.1145/3242969.3242985).

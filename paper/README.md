# Evaluation 2 paper package

Open `main.tex`. It is a complete IEEE-style two-column paper with the three original authors and registration numbers, all Evaluation 1 section categories, the implemented Evaluation 2 method, experimental setup, exact saved results, diagrams, real dashboard screenshots, limitations, future work and six academic references (including three journal papers).

The folder is self-contained for LaTeX compilation. The figure and generated-table files are included; access to WESAD or the trained models is not needed to compile the paper.

## Compile or edit

On Overleaf, upload the supplied source ZIP, select `main.tex` as the main document and use pdfLaTeX. The package uses standard IEEEtran, TikZ, graphicx and booktabs packages. References are inline in IEEE format, so no BibTeX run or separate `.bib` file is required.

With a local TeX installation, run from this folder:

```sh
latexmk -pdf main.tex
```

Alternatively:

```sh
pdflatex main.tex
pdflatex main.tex
```

The FISAC QP document refers to a prescribed LaTeX template, but a separate institution-provided `.tex` template was not supplied. This package follows the IEEE conference layout used by `EdgeBased_Healthcare.pdf`. If the faculty has a different mandatory template, transfer the content into it before submission. Add department/institution details if required; those details were not invented.

## Evaluation 1 sections retained and updated

| Evaluation 1 material | Evaluation 2 location and update |
| --- | --- |
| Abstract | Completed implementation, quantitative results and explicit benchmark distinction |
| IoT/AIoT introduction and architecture | Section I; application-specific architecture figure |
| Literature survey and comparative table | Section II; focused verified survey, including three journal papers and adaptive-sensing prior art |
| Research gap | Section III; calibrated quality-aware routing and joint evaluation |
| Problem statement | Section IV; baseline/stress/insufficient-evidence decision |
| Research objectives | Section V; objectives mapped to implemented experiments |
| Proposed AIoT methodology | Section VI; architecture, acquisition, preprocessing, features, experts, decision rules, events and cloud interaction |
| Implementation/simulation and evaluation | New Sections VII–VIII; completed software, baselines, ablations, all 11 scenarios, figures and screenshots |
| Limitations and ethical considerations | Section IX; measured weaknesses, dataset limits and research-use scope |
| Future scope | Section X; unimplemented CNN, live hardware, cost-aware selection, robust features and power measurement |
| Conclusion and IEEE references | Section XI and bibliography; obtained findings replace projected outcomes |

The original title/application and author identities are retained in substance. The method is now an ECG-first quality-aware multimodal cascade. CNN/KNN were candidates in the proposal and are not presented as completed experiments. The literature is revised around the implemented contribution; it does not repeat unverified claims about every paper in the earlier broad survey.

## Results and figures

- `generated/numbers.tex`, `clean_table.tex` and `failure_table.tex` come directly from the saved v2 comparison CSV.
- `figures/heldout_confusion.pdf` uses saved row-level clean predictions from the exact same 434 eligible windows.
- `figures/revision_tradeoff.pdf` includes favorable results and amplitude/dropout regressions.
- `figures/waveform_diagnostic.png` is a qualitative S9 diagnostic, not a held-out detector-validation result.
- The architecture and routing diagrams are editable TikZ drawings in `main.tex`.
- Four actual dashboard screenshots are included; the prediction and monitoring views appear in the paper. They show a functional S10 demonstration with the v2 artifact, not a held-out evaluation. `screenshots.json` records capture details. The browser uses a light theme; capture-only CSS hides local filesystem paths and corrects sidebar text contrast. No prediction or metric values were changed.
- Screenshot latency numbers illustrate individual UI runs and must not replace the controlled timing tables.
- `generated/assets_manifest.json` records figure hashes. `generated/tex_engine.json` records the temporary compiler used for the local build.

The 15.08% CPU saving is a held-out **training-procedure** result. The separately fitted dashboard artifact did not demonstrate CPU saving on its benchmark. The paper preserves that distinction, the unmet 25% target, amplitude-scaling regression, lower clean pooled F1 versus v1 and weak event-onset detection. Do not remove those qualifications while retaining the associated improvement claims.

## Regenerate assets in the project workspace

From the project root, with its Python environment:

```sh
.venv/bin/python paper/tools/build_assets.py
```

This script needs the existing saved reports and makes no model changes. Screenshot regeneration additionally requires the running local Streamlit app and headless Chrome; see `tools/capture_dashboard.mjs`. The screenshots already included are sufficient for compilation.

## Submission items that remain with the team

Check the faculty template, author/institution details and any prescribed page limit. FISAC QP also requires an institution-generated Turnitin similarity report and a five-minute contribution from each presenter. This package does not fabricate a similarity score, an institutional approval or a completed presentation. Review and disclose AI-assisted drafting according to your institution's rules.

The methods and results are suitable for an exploratory course research report. No physical edge device, live wearable system, energy experiment, independent clinical study or statistically proven superiority is claimed.

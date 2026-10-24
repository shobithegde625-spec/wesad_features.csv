"""Build paper figures and tables exclusively from saved experiment outputs."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
FIG = PAPER / "figures"
GEN = PAPER / "generated"
for directory in (FIG, GEN):
    directory.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PAPER / ".mplcache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PAPER / ".cache"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

CURRENT = ROOT / "reports/multimodal-real-v2"
comparison = pd.read_csv(CURRENT / "comparison.csv")
clean = comparison[comparison.scenario.eq("clean")].set_index("method")
cascade, fusion = clean.loc["cascade"], clean.loc["fusion"]
baseline = clean.loc["legacy_svm"]
numbers = {
    "CascadeAccuracy": f"{100*cascade.accuracy:.2f}",
    "CascadeRecall": f"{100*cascade.all_window_stress_recall:.2f}",
    "CascadeFone": f"{cascade.stress_f1:.3f}",
    "FusionRecall": f"{100*fusion.all_window_stress_recall:.2f}",
    "RecallGap": f"{100*(fusion.all_window_stress_recall-cascade.all_window_stress_recall):.2f}",
    "CpuSaving": f"{100*(1-cascade.mean_cpu_ms/fusion.mean_cpu_ms):.2f}",
    "CascadeCpu": f"{cascade.mean_cpu_ms:.2f}",
    "FusionCpu": f"{fusion.mean_cpu_ms:.2f}",
    "RecallGain": f"{100*(cascade.all_window_stress_recall-baseline.all_window_stress_recall):.2f}",
}
(GEN / "numbers.tex").write_text("% Generated from reports/multimodal-real-v2/comparison.csv\n" +
    "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k,v in numbers.items())+"\n")
names = {"legacy_logistic":"Original ECG logistic", "legacy_svm":"Original ECG SVM",
         "legacy_forest":"Original ECG forest", "ecg":"Physiological ECG",
         "confidence_only":"Confidence-only cascade", "cascade":r"\textbf{Quality-aware cascade}",
         "no_abstention":"Cascade without abstention", "fusion":"Always-on fusion"}
lines = [r"\begin{table*}[t]",r"\centering\caption{Clean participant-held-out results on the same 434 eligible replay windows. All-window recall counts rejected stress as undetected. Accuracy, precision and F1 describe accepted windows; all listed v2 methods have full clean coverage. Baseline CPU was not instrumented and is left unreported.}",
         r"\label{tab:clean}\small",r"\begin{tabular}{lrrrrrrr}\toprule",
         r"Method & Acc. (\%) & Bal. acc. (\%) & Prec. (\%) & Recall (\%) & F1 & Cov. (\%) & CPU (ms)\\\midrule"]
for method, label in names.items():
    row=clean.loc[method]
    cpu=f"{row.mean_cpu_ms:.2f}" if pd.notna(row.mean_cpu_ms) else "---"
    lines.append(f"{label} & {100*row.accuracy:.2f} & {100*row.balanced_accuracy:.2f} & {100*row.stress_precision:.2f} & {100*row.all_window_stress_recall:.2f} & {row.stress_f1:.3f} & {100*row.coverage:.1f} & {cpu}"+r"\\")
lines += [r"\bottomrule\end{tabular}",r"\end{table*}"]
(GEN / "clean_table.tex").write_text("\n".join(lines)+"\n")
scenario_names={"clean":"Clean", "ecg_missing_burst":"ECG missing burst", "ecg_flatline":"ECG flatline", "ecg_clipping":"ECG clipping", "ecg_noise_moderate":"Moderate ECG noise", "ecg_noise_severe":"Severe ECG noise", "ecg_amplitude_scale":r"ECG amplitude $\times3$", "eda_dropout":"EDA dropout", "resp_dropout":"Respiration dropout", "alternate_dropout":"Both alternates dropped", "total_dropout":"Total sensor dropout"}
lines=[r"\begin{table*}[t]",r"\centering\caption{All eleven waveform scenarios: quality-aware cascade compared with the same fold-specific, full-modality-required fusion expert. Acc. and F1 use accepted decisions; recall uses all stress windows. A dash indicates undefined classification quality when coverage is zero. Each scenario has 434 scored windows.}",
       r"\label{tab:failures}\small",r"\begin{tabular}{lrrrrrrrr}\toprule",
       r" & \multicolumn{4}{c}{Adaptive cascade} & \multicolumn{4}{c}{Always-on fusion}\\",
       r"Scenario & Cov. (\%) & Acc. (\%) & Rec. (\%) & F1 & Cov. (\%) & Acc. (\%) & Rec. (\%) & F1\\\midrule"]
for scenario,label in scenario_names.items():
    fields=[]
    for method in ("cascade","fusion"):
        row=comparison[(comparison.scenario==scenario)&(comparison.method==method)].iloc[0]
        for field in ("coverage","accuracy","all_window_stress_recall","stress_f1"):
            value=row[field]
            fields.append("---" if pd.isna(value) else f"{value:.3f}" if field=="stress_f1" else f"{100*value:.2f}")
    lines.append(label+" & "+" & ".join(fields)+r"\\")
lines += [r"\bottomrule\end{tabular}",r"\end{table*}"]
(GEN / "failure_table.tex").write_text("\n".join(lines)+"\n")

cols=["method","scenario","label","prediction","accepted"]
pred=pd.read_csv(CURRENT / "predictions.csv",usecols=cols)
fig,axes=plt.subplots(1,2,figsize=(8,3.4),layout="constrained")
for ax,method,title in zip(axes,("legacy_svm","cascade"),("Original-feature ECG SVM","Adaptive multimodal cascade")):
    group=pred[(pred.method==method)&pred.scenario.eq("clean")&pred.label.isin([0,1])]
    assert group.accepted.all() and len(group)==434
    cm=confusion_matrix(group.label.astype(int),group.prediction.astype(int),labels=[0,1])
    ax.imshow(cm,cmap="Blues",vmin=0,vmax=282)
    for (i,j),value in np.ndenumerate(cm):
        ax.text(j,i,str(value),ha="center",va="center",fontsize=15,color="white" if value>141 else "#123745")
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["Baseline","Stress"],yticklabels=["Baseline","Stress"],xlabel="Predicted",ylabel="Reference",title=title)
fig.savefig(FIG / "heldout_confusion.pdf",bbox_inches="tight")
fig.savefig(FIG / "heldout_confusion.png",dpi=180,bbox_inches="tight")
plt.close(fig)
for source,name in [(ROOT / "reports/multimodal-revision-comparison/revision_tradeoff.pdf","revision_tradeoff.pdf"),
                    (ROOT / "reports/multimodal-diagnostics/S9-baseline-waveforms.png","waveform_diagnostic.png")]:
    shutil.copy2(source,FIG/name)
manifest={"source":"Saved experiments; no retraining or operating-point selection", "figures":{}}
for p in sorted(FIG.glob("*")):
    if p.is_file() and p.suffix in (".png",".pdf"):
        manifest["figures"][p.name]={"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"bytes":p.stat().st_size}
(GEN / "assets_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print("Generated exact result tables, number macros, confusion matrices and copied source figures.")

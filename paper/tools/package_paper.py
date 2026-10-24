"""Package the self-contained manuscript without caches, raw data or TeX auxiliaries."""
from pathlib import Path
import hashlib
import json
import re
import zipfile

root = Path(__file__).resolve().parents[2]
paper = root / "paper"
source = (paper / "main.tex").read_text()
generated = "\n".join(p.read_text() for p in (paper / "generated").glob("*.tex"))
labels = re.findall(r"\\label\{([^}]+)\}", source + generated)
references = re.findall(r"\\ref\{([^}]+)\}", source)
citations = [k for group in re.findall(r"\\cite\{([^}]+)\}", source) for k in group.split(",")]
bibliography = re.findall(r"\\bibitem\{([^}]+)\}", source)
assert len(labels) == len(set(labels)), "Duplicate labels"
assert not set(references).difference(labels), "Unresolved figure/table references"
assert not set(citations).difference(bibliography), "Unresolved citations"
assert (paper / "main.pdf").read_bytes().startswith(b"%PDF-"), "Compile main.tex first"
figures = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", source)
assert all((paper / "figures" / name).exists() for name in figures), "Missing figure"
files = [paper / name for name in ("main.tex", "main.pdf", "README.md", "SOURCE_NOTES.md")]
for directory in ("figures", "generated", "tools"):
    files.extend(p for p in sorted((paper / directory).glob("*")) if p.is_file() and p.suffix in (".tex", ".png", ".pdf", ".json", ".py", ".mjs"))
archive = root / "evaluation2_paper_overleaf.zip"
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as package:
    for file in files:
        package.write(file, file.relative_to(paper).as_posix())
with zipfile.ZipFile(archive) as package:
    assert package.testzip() is None
    assert "main.tex" in package.namelist()
record = dict(archive=str(archive.name), files=len(files), bytes=archive.stat().st_size,
              sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
              references=len(bibliography), unique_labels=len(labels),
              unresolved_citations=0, unresolved_references=0, missing_figures=0)
(paper / "package_manifest.json").write_text(json.dumps(record, indent=2)+"\n")
print(json.dumps(record, indent=2))

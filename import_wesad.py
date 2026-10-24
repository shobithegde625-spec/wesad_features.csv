"""Verify the official archive and extract synchronized recordings only.

Download separately with resumable curl from the official university share.
This importer preserves existing files and excludes redundant raw text exports.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import zipfile

OFFICIAL_ARCHIVE_URL = "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download/WESAD.zip"
OFFICIAL_PAGE = "https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html"
OFFICIAL_BYTES = 2249444501
OFFICIAL_SHA1 = "bae786bf599ef28e42e1c2e18da9b84d685e84e7"
EXPECTED_SUBJECTS = {f"S{i}" for i in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17)}


def digest_file(path, algorithms=("sha256", "sha1")):
    hashes = {name: hashlib.new(name) for name in algorithms}
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b""):
            for digest in hashes.values():
                digest.update(chunk)
    return {name: digest.hexdigest() for name, digest in hashes.items()}


def selected_members(archive):
    selected = []
    subjects = set()
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if member.is_dir() or "__MACOSX" in path.parts:
            continue
        if path.is_absolute() or ".." in path.parts or "\\" in member.filename:
            raise ValueError(f"Unsafe archive member: {member.filename}")
        match = re.search(r"(?:^|/)(S\d+)/\1\.pkl$", member.filename)
        if match:
            subject = match[1]
            if subject in subjects:
                raise ValueError(f"Duplicate synchronized recording for {subject}")
            subjects.add(subject)
            selected.append((member, Path(subject) / f"{subject}.pkl"))
        elif "readme" in path.name.lower() or "license" in path.name.lower():
            if path.suffix.lower() in (".pdf", ".txt", ".md"):
                relative = Path(*path.parts[-2:]) if len(path.parts) >= 2 and re.fullmatch(r"S\d+", path.parts[-2]) else Path(path.name)
                selected.append((member, relative))
    if subjects != EXPECTED_SUBJECTS:
        raise ValueError(f"Unexpected participant set: {sorted(subjects)}")
    return selected


def import_archive(archive_path, output, verify_official=True, archive_only=False):
    archive_path, output = Path(archive_path), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing raw directory: {output}")
    if verify_official and archive_path.stat().st_size != OFFICIAL_BYTES:
        raise ValueError("Official archive byte count does not match; download may be incomplete")
    print("Verifying archive checksums...", flush=True)
    hashes = digest_file(archive_path)
    if verify_official and hashes["sha1"] != OFFICIAL_SHA1:
        raise ValueError("Archive SHA-1 does not match the checksum supplied by the official host")
    with zipfile.ZipFile(archive_path) as archive:
        selected = selected_members(archive)
        recording_index = [{"archive_member": member.filename, "bytes": member.file_size,
                            "compressed_bytes": member.compress_size, "archive_crc32": f"{member.CRC:08x}"}
                           for member, relative in selected if relative.suffix == ".pkl"]
        if archive_only:
            selected = [(member, relative) for member, relative in selected if relative.suffix != ".pkl"]
        required = sum(member.file_size for member, _ in selected)
        available = shutil.disk_usage(output.parent if output.parent.exists() else Path.cwd()).free
        print(f"Selected {len(selected)} files; {required / 1024**3:.2f} GiB; free {available / 1024**3:.2f} GiB", flush=True)
        if required + 512*1024**2 > available:
            raise OSError("Insufficient disk space for selected recordings plus a 512 MiB reserve")
        output.mkdir(parents=True, exist_ok=False)
        files = []
        for member, relative in selected:
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            print(f"Extracting {relative} ({member.file_size/1024**2:.1f} MiB)", flush=True)
            with archive.open(member) as source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination, length=8*1024*1024)
            files.append({"path": str(relative), "bytes": target.stat().st_size,
                          "sha256": digest_file(target, ("sha256",))["sha256"], "archive_crc32": f"{member.CRC:08x}"})
    provenance = {"source_page": OFFICIAL_PAGE, "download_url": OFFICIAL_ARCHIVE_URL,
                  "archive_bytes": archive_path.stat().st_size, "archive_hashes": hashes,
                  "official_checksum_verified": verify_official, "files": files,
                  "compressed_access": archive_only, "archive_path": str(archive_path.resolve()),
                  "recording_index": recording_index,
                  "acquired_at": datetime.now(timezone.utc).isoformat(),
                  "usage_terms": "Scientific, non-commercial use with credit to the dataset owners; see source page.",
                  "citation": "Schmidt, Reiss, Duerichen, Marberger and Van Laerhoven (2018). Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection. ICMI. DOI: 10.1145/3242969.3242985."}
    (output / "acquisition.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Verified and extracted {len(files)} files to {output}", flush=True)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive-only", action="store_true", help="Verify archive and extract only documentation; read recordings directly from ZIP")
    args = parser.parse_args()
    try:
        import_archive(args.archive, args.output, archive_only=args.archive_only)
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

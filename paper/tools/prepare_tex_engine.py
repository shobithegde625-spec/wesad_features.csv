"""Download the official Tectonic binary into a temporary, isolated directory."""
from pathlib import Path
import hashlib
import io
import json
import platform
import tarfile
import requests

destination = Path("/private/tmp/pulsepoint-tectonic")
destination.mkdir(exist_ok=True)
api = "https://api.github.com/repos/tectonic-typesetting/tectonic/releases/latest"
response = requests.get(api, timeout=30)
response.raise_for_status()
release = response.json()
target = "aarch64-apple-darwin" if platform.machine() == "arm64" else "x86_64-apple-darwin"
asset = next(a for a in release["assets"] if target in a["name"] and a["name"].endswith(".tar.gz"))
download = requests.get(asset["browser_download_url"], timeout=120)
download.raise_for_status()
digest = hashlib.sha256(download.content).hexdigest()
if asset.get("digest"):
    assert asset["digest"] == "sha256:" + digest, "Official release digest mismatch"
with tarfile.open(fileobj=io.BytesIO(download.content), mode="r:gz") as archive:
    member = next(m for m in archive.getmembers() if Path(m.name).name == "tectonic" and m.isfile())
    binary = destination / "tectonic"
    binary.write_bytes(archive.extractfile(member).read())
    binary.chmod(0o700)
record = dict(release=release["tag_name"], asset=asset["name"], url=asset["browser_download_url"], sha256=digest,
              official_digest_verified=bool(asset.get("digest")), installed_to=str(binary))
Path("paper/generated/tex_engine.json").write_text(json.dumps(record, indent=2)+"\n")
print("Prepared", binary, release["tag_name"])

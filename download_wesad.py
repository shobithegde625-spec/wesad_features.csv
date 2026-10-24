"""Resume the official WESAD archive with four bounded HTTP range requests."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
from pathlib import Path
import shutil
import time

import requests
from import_wesad import OFFICIAL_ARCHIVE_URL, OFFICIAL_BYTES, OFFICIAL_SHA1, digest_file


def download(output, prefix_file=None, workers=4):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if not 1 <= workers <= 4:
        raise ValueError("Use one to four workers")
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = Path(prefix_file) if prefix_file else None
    prefix_bytes = prefix.stat().st_size if prefix and prefix.exists() else 0
    if prefix_bytes > OFFICIAL_BYTES:
        raise ValueError("Prefix is larger than the official archive")
    if prefix_bytes:
        with prefix.open("rb") as handle:
            if handle.read(4) != b"PK\x03\x04":
                raise ValueError("Existing prefix does not start with a ZIP header")
    temporary = output.with_name(output.name + ".chunks")
    temporary.mkdir(exist_ok=True)
    needed = OFFICIAL_BYTES-prefix_bytes+OFFICIAL_BYTES+512*1024**2
    if shutil.disk_usage(output.parent).free < needed:
        raise OSError("Insufficient free space for resumable chunks, assembled archive and a 512 MiB reserve")
    ranges = []
    width = math.ceil((OFFICIAL_BYTES-prefix_bytes)/workers) if prefix_bytes < OFFICIAL_BYTES else 0
    for index in range(workers):
        start = prefix_bytes+index*width
        end = min(OFFICIAL_BYTES, start+width)-1
        if start <= end:
            ranges.append((start, end, temporary / f"{start}-{end}.part"))

    def fetch(task):
        base, end, path = task
        for attempt in range(4):
            existing = path.stat().st_size if path.exists() else 0
            start = base+existing
            if start == end+1:
                return path
            if start > end+1:
                raise ValueError(f"Unexpected range chunk length: {path}")
            try:
                with requests.get(OFFICIAL_ARCHIVE_URL, headers={"Range": f"bytes={start}-{end}"},
                                  stream=True, timeout=(20, 60)) as response:
                    response.raise_for_status()
                    expected = f"bytes {start}-{end}/{OFFICIAL_BYTES}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("Official server did not honor the exact requested byte range")
                    written = existing
                    next_notice = (written//(32*1024**2)+1)*32*1024**2
                    with path.open("ab") as handle:
                        for block in response.iter_content(1024*1024):
                            if block:
                                handle.write(block)
                                written += len(block)
                                if written >= next_notice:
                                    print(f"Range {base}-{end}: {written/1024**2:.0f}/{(end-base+1)/1024**2:.0f} MiB", flush=True)
                                    next_notice += 32*1024**2
                if path.stat().st_size != end-base+1:
                    raise OSError("Range response ended before all requested bytes arrived")
                print(f"Range {base}-{end} complete", flush=True)
                return path
            except (requests.RequestException, OSError) as exc:
                if attempt == 3:
                    raise
                print(f"Range retry {attempt+1}: {exc}", flush=True)
                time.sleep(2*(attempt+1))
        raise RuntimeError("Unreachable")

    print(f"Resuming {prefix_bytes/1024**2:.1f} MiB; fetching {(OFFICIAL_BYTES-prefix_bytes)/1024**2:.1f} MiB in {len(ranges)} ranges", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, task) for task in ranges]
        for future in as_completed(futures):
            future.result()
    assembled = output.with_name(output.name + ".assembling")
    if assembled.exists():
        raise FileExistsError(f"A prior assembled file exists: {assembled}")
    print("Assembling and verifying the official archive...", flush=True)
    parts = ([prefix] if prefix_bytes else []) + [task[2] for task in ranges]
    with assembled.open("xb") as destination:
        for path in parts:
            with path.open("rb") as source:
                shutil.copyfileobj(source, destination, length=8*1024*1024)
    hashes = digest_file(assembled)
    if assembled.stat().st_size != OFFICIAL_BYTES or hashes["sha1"] != OFFICIAL_SHA1:
        raise ValueError("Assembled archive does not match the official checksum; chunks are preserved for inspection")
    assembled.rename(output)
    # These are only the temporary chunks explicitly consumed above, never user raw files.
    for path in parts:
        path.unlink()
    if not any(temporary.iterdir()):
        temporary.rmdir()
    print(f"Verified {output}; SHA-256 {hashes['sha256']}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/downloads/WESAD.zip"))
    parser.add_argument("--prefix-file", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    try:
        download(args.output, args.prefix_file, args.workers)
    except (ValueError, OSError, requests.RequestException) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

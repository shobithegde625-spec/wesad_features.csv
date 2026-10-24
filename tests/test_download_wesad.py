import hashlib
from pathlib import Path
import re
import pytest
import download_wesad as module


def mock_archive(monkeypatch, payload, honor_range=True):
    monkeypatch.setattr(module, "OFFICIAL_BYTES", len(payload))
    monkeypatch.setattr(module, "OFFICIAL_SHA1", hashlib.sha1(payload).hexdigest())
    calls = []
    class Response:
        status_code = 206 if honor_range else 200
        def __init__(self, start, end):
            self.start, self.end = start, end
            self.headers = {"Content-Range": f"bytes {start}-{end}/{len(payload)}"}
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def raise_for_status(self): pass
        def iter_content(self, size):
            yield payload[self.start:self.end+1]
    def get(url, headers, **kwargs):
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", headers["Range"])
        start, end = map(int, match.groups())
        calls.append((start, end))
        return Response(start, end)
    monkeypatch.setattr(module.requests, "get", get)
    return calls


def test_parallel_download_reuses_prefix_and_verifies_before_cleanup(monkeypatch, tmp_path):
    payload = b"PK\x03\x04" + bytes(range(256))*8
    calls = mock_archive(monkeypatch, payload)
    prefix = tmp_path / "prefix.part"
    prefix.write_bytes(payload[:128])
    output = tmp_path / "WESAD.zip"
    module.download(output, prefix, workers=4)
    assert output.read_bytes() == payload
    assert not prefix.exists()
    assert min(start for start, _ in calls) == 128
    assert max(end for _, end in calls) == len(payload)-1
    assert not output.with_name(output.name+".chunks").exists()


def test_unhonored_range_never_produces_an_archive(monkeypatch, tmp_path):
    payload = b"PK\x03\x04"+b"data"*100
    mock_archive(monkeypatch, payload, honor_range=False)
    output = tmp_path / "WESAD.zip"
    with pytest.raises(ValueError, match="byte range"):
        module.download(output, workers=1)
    assert not output.exists()


def test_bad_checksum_preserves_download_evidence(monkeypatch, tmp_path):
    payload = b"PK\x03\x04"+b"data"*100
    mock_archive(monkeypatch, payload)
    monkeypatch.setattr(module, "OFFICIAL_SHA1", "0"*40)
    output = tmp_path / "WESAD.zip"
    with pytest.raises(ValueError, match="checksum"):
        module.download(output, workers=1)
    assert not output.exists()
    assert output.with_name(output.name+".assembling").exists()
    assert list(output.with_name(output.name+".chunks").glob("*.part"))

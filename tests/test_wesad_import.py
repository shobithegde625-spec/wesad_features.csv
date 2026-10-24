import hashlib
import json
from pathlib import Path
import zipfile
import pytest
from import_wesad import EXPECTED_SUBJECTS, import_archive, selected_members


def test_archive_only_import_preserves_compressed_recordings(tmp_path):
    archive = tmp_path / "WESAD.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for subject in EXPECTED_SUBJECTS:
            handle.writestr(f"WESAD/{subject}/{subject}.pkl", b"test-recording")
        handle.writestr("WESAD/wesad_readme.pdf", b"test-document")
        handle.writestr("WESAD/S2/S2_respiban.txt", b"redundant-export")
    output = tmp_path / "info"
    provenance = import_archive(archive, output, verify_official=False, archive_only=True)
    assert provenance["compressed_access"]
    assert len(provenance["recording_index"]) == 15
    assert not list(output.glob("**/*.pkl"))
    assert (output / "wesad_readme.pdf").read_bytes() == b"test-document"
    assert provenance["archive_hashes"]["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert not provenance["official_checksum_verified"]
    with pytest.raises(FileExistsError):
        import_archive(archive, output, verify_official=False)


def test_incomplete_official_archive_is_rejected_without_output(tmp_path):
    archive = tmp_path / "incomplete.zip"
    archive.write_bytes(b"partial")
    output = tmp_path / "info"
    with pytest.raises(ValueError, match="incomplete"):
        import_archive(archive, output)
    assert not output.exists()

"""Tests for reproducible MetroPT-3 ingestion helpers."""

import hashlib
from pathlib import Path
from zipfile import ZipFile

from machinepulse.data.ingest import extract_zip_member, sha256_file


def test_extract_zip_member_and_hash(tmp_path: Path) -> None:
    """The configured dataset member is extracted without flattening the archive."""
    archive_path = tmp_path / "dataset.zip"
    output_path = tmp_path / "raw" / "MetroPT3(AirCompressor).csv"
    content = b"timestamp,TP2\n2020-01-01 00:00:00,1.2\n"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("MetroPT3/MetroPT3(AirCompressor).csv", content)

    member = extract_zip_member(archive_path, output_path.name, output_path)

    assert member == "MetroPT3/MetroPT3(AirCompressor).csv"
    assert output_path.read_bytes() == content
    assert sha256_file(output_path) == hashlib.sha256(content).hexdigest()

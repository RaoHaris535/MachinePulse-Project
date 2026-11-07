"""Download and extract the MetroPT-3 dataset from its official UCI source."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import BadZipFile, ZipFile

import requests

from machinepulse.config import load_yaml


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(
    url: str,
    destination: Path,
    *,
    timeout_seconds: int,
    chunk_bytes: int,
) -> None:
    """Stream a remote archive to a temporary local path."""
    with requests.get(url, stream=True, timeout=timeout_seconds) as response:
        response.raise_for_status()
        with destination.open("wb") as stream:
            for chunk in response.iter_content(chunk_size=chunk_bytes):
                if chunk:
                    stream.write(chunk)


def extract_zip_member(archive_path: Path, expected_name: str, output_path: Path) -> str:
    """Extract exactly one named member and return its full archive member name."""
    try:
        with ZipFile(archive_path) as archive:
            corrupt_member = archive.testzip()
            if corrupt_member is not None:
                message = f"Corrupt member in downloaded archive: {corrupt_member}"
                raise ValueError(message)

            matches = [
                name
                for name in archive.namelist()
                if PurePosixPath(name).name == expected_name and not name.endswith("/")
            ]
            if len(matches) != 1:
                message = (
                    f"Expected exactly one archive member named {expected_name!r}; "
                    f"found {len(matches)}"
                )
                raise ValueError(message)

            member_name = matches[0]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_output = output_path.with_suffix(f"{output_path.suffix}.part")
            try:
                with archive.open(member_name) as source, temporary_output.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                temporary_output.replace(output_path)
            finally:
                temporary_output.unlink(missing_ok=True)
    except BadZipFile as error:
        message = f"Downloaded file is not a valid ZIP archive: {archive_path}"
        raise ValueError(message) from error

    return member_name


def _existing_data_is_current(raw_path: Path, manifest_path: Path) -> bool:
    if not raw_path.is_file() or not manifest_path.is_file():
        return False

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False

    expected_digest = manifest.get("extracted_sha256")
    expected_bytes = manifest.get("extracted_bytes")
    return (
        isinstance(expected_digest, str)
        and expected_bytes == raw_path.stat().st_size
        and expected_digest == sha256_file(raw_path)
    )


def ingest(config: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    """Materialize the configured raw dataset and a deterministic source manifest."""
    source = config["source"]
    raw_path = Path(source["raw_path"])
    manifest_path = Path(source["manifest_path"])

    if not force and _existing_data_is_current(raw_path, manifest_path):
        print(f"Raw dataset is already present and verified: {raw_path}")
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="metropt3-", dir=raw_path.parent) as temp_dir:
        archive_path = Path(temp_dir) / "metropt3.zip"
        print(f"Downloading MetroPT-3 from {source['url']}")
        download_archive(
            source["url"],
            archive_path,
            timeout_seconds=int(source["request_timeout_seconds"]),
            chunk_bytes=int(source["download_chunk_bytes"]),
        )
        archive_digest = sha256_file(archive_path)
        member_name = extract_zip_member(
            archive_path,
            source["archive_member"],
            raw_path,
        )

    manifest = {
        "source_url": source["url"],
        "archive_member": member_name,
        "archive_sha256": archive_digest,
        "extracted_path": raw_path.as_posix(),
        "extracted_bytes": raw_path.stat().st_size,
        "extracted_sha256": sha256_file(raw_path),
    }
    manifest_path.write_text(
        f"{json.dumps(manifest, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    print(f"Wrote {raw_path} ({manifest['extracted_bytes']:,} bytes)")
    print(f"Wrote source manifest to {manifest_path}")
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml", help="Data YAML path")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even when the existing file matches its manifest",
    )
    return parser.parse_args()


def main() -> None:
    """Run the MetroPT-3 ingestion command."""
    args = parse_args()
    ingest(load_yaml(args.config), force=args.force)


if __name__ == "__main__":
    main()

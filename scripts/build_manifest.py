"""Record the provenance of every raw source file.

Writes data/raw/manifest.json holding, per file: the download URL, a SHA-256
hash, the byte size, and the publication and data cut dates read from the
file's own config sheet. Where config.yaml records an expected hash, a
mismatch stops the run.

Dataset identity is taken from the URL and the hash rather than the filename
or cover sheet. The cover sheet of the daily incident file reads "Low level
geography dataset".

Usage:
    uv run python scripts/build_manifest.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdh.config import load_config  # noqa: E402
from fdh.contract import CONFIG_SHEET_NAMES  # noqa: E402
from fdh.ods import read_config_sheet  # noqa: E402

CHUNK = 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    config = load_config()
    raw_dir = config.paths.resolved("raw_dir")
    if not raw_dir.exists():
        raise SystemExit(f"raw directory does not exist: {raw_dir}")

    entries = []
    failures = []

    for collection, sources in config.sources.items():
        for source in sources:
            path = raw_dir / source.filename
            if not path.exists():
                failures.append(f"missing file: {path}")
                continue

            digest = sha256(path)
            if source.sha256 and digest != source.sha256:
                failures.append(
                    f"{source.filename}: expected sha256 {source.sha256}, found {digest}. "
                    f"The file on disk is a different release from the one config.yaml records."
                )
                continue

            try:
                sheet_values = read_config_sheet(path, CONFIG_SHEET_NAMES)
            except Exception as error:  # config sheet may be absent in older eras
                failures.append(f"{source.filename}: {error}")
                sheet_values = {}

            entries.append(
                {
                    "collection": collection,
                    "era": source.era,
                    "filename": source.filename,
                    "url": source.url,
                    "sha256": digest,
                    "bytes": path.stat().st_size,
                    "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "file_config_sheet": sheet_values,
                }
            )
            status = "verified" if source.sha256 else "recorded"
            print(f"{status}: {source.filename}  {digest}")
            for key, value in sheet_values.items():
                print(f"    {key}: {value}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1

    manifest_path = raw_dir / "manifest.json"
    manifest_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"\nwrote {manifest_path} with {len(entries)} entries")
    print("Copy each new sha256 into config.yaml so later runs verify rather than record.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
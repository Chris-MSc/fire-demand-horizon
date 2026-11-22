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

import json
import sys
from datetime import UTC, datetime

from fdh.config import INCIDENT_COLLECTION, incident_sources, load_config
from fdh.contract import CONFIG_SHEET_NAMES
from fdh.ods import read_config_sheet
from fdh.provenance import sha256_file


def main() -> int:
    config = load_config()
    raw_dir = config.paths.resolved("raw_dir")
    if not raw_dir.exists():
        raise SystemExit(f"raw directory does not exist: {raw_dir}")

    entries = []
    failures = []

    # Incident files only. The boundary layers are not ODS and have no config
    # sheet; their provenance record comes from fdh.boundaries.save_layer.
    for source in incident_sources(config):
        path = raw_dir / source.filename
        if not path.exists():
            failures.append(f"missing file: {path}")
            continue

        digest = sha256_file(path)
        if source.sha256 and digest != source.sha256:
            failures.append(
                f"{source.filename}: expected sha256 {source.sha256}, found {digest}. "
                f"The file on disk is a different release from the one config.yaml records."
            )
            continue

        sheet_values = read_config_sheet(path, CONFIG_SHEET_NAMES)
        entries.append(
            {
                "collection": INCIDENT_COLLECTION,
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
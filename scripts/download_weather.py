"""Download the configured span of the HadUK-Grid daily archive, resumably.

Lists the archive for weather.variables across weather.window_start to
weather.release_covers_to, reports the file count and total size, then fetches
what is not already on disk into data/raw/weather/<variable>/. A file on disk
at the listed size is skipped and one at another size is fetched again, so a
run with no arguments resumes an earlier one.

A failed file is retried twice and then recorded, and the run carries on.
Three consecutive authentication failures end the run, since an expired token
fails every file after it.

The token is read from CEDA_TOKEN, or from CEDA_ACCESS_TOKEN, or prompted for.
It is held in this process only and is written neither to disk nor to the
record.

Writes download_record.json beside the variable directories.

Usage:
    uv run python scripts/download_weather.py
    uv run python scripts/download_weather.py --dry-run
    uv run python scripts/download_weather.py --limit 1
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import requests

from fdh import weather as wx
from fdh.config import load_config, require

RECORD_FILE = "download_record.json"
TOKEN_VARIABLES = ("CEDA_TOKEN", "CEDA_ACCESS_TOKEN")


def set_token() -> bool:
    """Place the token where fdh.weather reads it, prompting if neither variable is set."""
    found = next((os.environ[name] for name in TOKEN_VARIABLES if os.environ.get(name)), None)
    if not found:
        found = getpass.getpass("CEDA token: ")
    os.environ["CEDA_ACCESS_TOKEN"] = found.strip()
    return bool(os.environ["CEDA_ACCESS_TOKEN"])


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def write_record(path: Path, settings: dict, run: wx.DownloadRun, started: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {**settings, **asdict(run), "started_at": started, "finished_at": now()}
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Download only the first N files per variable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List and size the files without downloading.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    config = load_config()
    weather = config.weather
    variables = require(weather.variables, "weather.variables")
    base_url = require(weather.base_url, "weather.base_url")
    version = require(weather.version_directory, "weather.version_directory")
    destination = config.paths.resolved("raw_dir") / "weather"

    print(f"Listing {base_url}")
    print(f"  span:      {weather.window_start} to {weather.release_covers_to}")
    print(f"  variables: {variables}")
    try:
        files = wx.collect(
            base_url,
            variables,
            version,
            weather.window_start,
            weather.release_covers_to,
            limit=args.limit,
        )
    except (wx.CedaError, requests.RequestException) as error:
        print(f"\nListing stopped.\n{error}", file=sys.stderr)
        return 1

    planned = wx.plan_downloads(files, destination)
    total = sum(item.file.size for item in planned)
    held = [item for item in planned if item.state == "present"]
    wrong = [item for item in planned if item.state == "wrong size"]
    to_fetch = [item for item in planned if item.state != "present"]

    for name, items in files.items():
        print(
            f"  {name}: {len(items)} files, {wx.total_bytes(items) / 1e9:.1f} GB, "
            f"{items[0].start} to {items[-1].end}"
        )
    if args.limit is not None:
        print(f"  limited to the first {args.limit} per variable")
    print(f"Listed {len(planned)} files, {total / 1e9:.1f} GB")
    print(
        f"  on disk at the listed size: {len(held)} files, "
        f"{sum(item.file.size for item in held) / 1e9:.1f} GB"
    )
    print(f"  on disk at another size, fetched again: {len(wrong)}")
    print(
        f"  to download: {len(to_fetch)} files, "
        f"{sum(item.file.size for item in to_fetch) / 1e9:.1f} GB, into {destination}"
    )

    if args.dry_run:
        print("Dry run: nothing downloaded and no record written.")
        return 0

    if to_fetch and not set_token():
        print("\nDownload stopped.\nNo CEDA token was given.", file=sys.stderr)
        return 1

    settings = {
        "span_start": weather.window_start.isoformat(),
        "span_end": weather.release_covers_to.isoformat(),
        "variables": variables,
        "version_directory": version,
        "base_url": base_url,
        "limit_per_variable": args.limit,
    }
    record_path = destination / RECORD_FILE
    started = now()
    run = wx.DownloadRun()
    try:
        wx.run_downloads(planned, run=run, report=lambda line: print(line, flush=True))
    except KeyboardInterrupt:
        run.stopped = "interrupted at the keyboard"
    finally:
        write_record(record_path, settings, run, started)

    print(f"\nFiles expected:            {run.files_expected}")
    print(f"  present:                 {run.files_present} ({run.bytes_present / 1e9:.1f} GB)")
    print(f"  skipped, already held:   {run.files_skipped}")
    print(
        f"  downloaded this run:     {run.files_downloaded} "
        f"({run.bytes_downloaded / 1e9:.1f} GB)"
    )
    print(f"  failed:                  {len(run.failures)}")
    for failure in run.failures:
        print(f"    {failure['file']}: {failure['reason']}")
    print(f"Wrote the download record to {record_path}")

    if run.stopped:
        print(f"\nDownload stopped: {run.stopped}.", file=sys.stderr)
        if "authentication" in run.stopped:
            print(
                "The token has probably expired. Running the script again with a "
                "new token resumes from where this run stopped.",
                file=sys.stderr,
            )
        return 1
    if run.failures:
        print("Running the script again fetches the files that failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

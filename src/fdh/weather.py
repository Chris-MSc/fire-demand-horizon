"""Access to HadUK-Grid daily files on the CEDA archive.

Directory listings under dap.ceda.ac.uk are public and carry file sizes;
file downloads redirect to a login page unless a CEDA access token is sent as
a bearer header. Tokens come from https://services.ceda.ac.uk/account/token/.

Daily files hold one month each, named
<variable>_hadukgrid_uk_1km_day_YYYYMMDD-YYYYMMDD.nc. A listing entry that does
not match that pattern stops the run.

The bulk download is resumable. A file on disk at the listed size is skipped,
a file at another size is removed and fetched again, a failed file is retried
and then recorded without ending the run, and a run of authentication failures
ends it, since an expired token fails every file after it.
"""

from __future__ import annotations

import getpass
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import requests

LISTING_ROW = re.compile(
    r'href="(?P<name>[^"]+)"[^<]*</a>\s+(?P<modified>\S+\s+\S+)\s+(?P<size>\d+)'
)
DIRECTORY = re.compile(r'href="(?P<name>[^"/]+/)"')
DAILY_FILE = re.compile(
    r"^(?P<variable>\w+)_hadukgrid_uk_1km_day_(?P<start>\d{8})-(?P<end>\d{8})\.nc$"
)
TIMEOUT = 600
CHUNK = 1024 * 1024

# Suffix of a file still being received. It is renamed once the byte count
# matches the listing, so a name ending in .nc is always a whole file.
PARTIAL_SUFFIX = ".part"

# Further attempts at a file after the first fails, and the pause between them.
RETRIES = 2
PAUSE_SECONDS = 10.0

# Consecutive files refused for authentication before the run ends.
AUTH_FAILURE_LIMIT = 3

AUTH_STATUS = (401, 403)


class CedaError(RuntimeError):
    """Raised when the archive does not respond as expected."""


class CedaAuthError(CedaError):
    """Raised when the archive refuses a request for want of a valid token."""


@dataclass(frozen=True)
class ArchiveFile:
    name: str
    url: str
    size: int
    start: date
    end: date


def token() -> str | None:
    return os.environ.get("CEDA_ACCESS_TOKEN")


def ensure_token() -> bool:
    """Prompt for a CEDA access token if CEDA_ACCESS_TOKEN is not set.

    The token is held in this process's environment only, never written to
    disk. Returns whether a token is now set.
    """
    if not token():
        os.environ["CEDA_ACCESS_TOKEN"] = getpass.getpass("CEDA token: ").strip()
    return bool(token())


def _get(url: str, **kwargs) -> requests.Response:
    headers = kwargs.pop("headers", {})
    access = token()
    if access:
        headers["Authorization"] = f"Bearer {access}"
    return requests.get(url, headers=headers, timeout=TIMEOUT, **kwargs)


def list_directories(url: str) -> list[str]:
    response = _get(url)
    response.raise_for_status()
    return [
        name
        for name in DIRECTORY.findall(response.text)
        if not name.startswith((".", "http"))
    ]


def list_files(url: str) -> list[tuple[str, int]]:
    response = _get(url)
    response.raise_for_status()
    return [
        (match.group("name"), int(match.group("size")))
        for match in LISTING_ROW.finditer(response.text)
        if match.group("name").endswith(".nc")
    ]


def daily_variables(base_url: str) -> dict[str, bool]:
    """Return every variable under the archive root and whether it publishes daily files."""
    root = base_url.rstrip("/")
    return {
        name.rstrip("/"): "day/" in list_directories(f"{root}/{name}")
        for name in list_directories(f"{root}/")
    }


def daily_files(
    base_url: str, variable: str, version: str, start: date, end: date
) -> list[ArchiveFile]:
    """Return the monthly files for one variable that overlap the date range."""
    directory = f"{base_url.rstrip('/')}/{variable}/day/{version}/"
    selected = []
    for name, size in list_files(directory):
        match = DAILY_FILE.match(name)
        if not match:
            raise CedaError(f"unexpected filename in {directory}: {name}")
        first = date.fromisoformat(match.group("start"))
        last = date.fromisoformat(match.group("end"))
        if last >= start and first <= end:
            selected.append(
                ArchiveFile(name=name, url=directory + name, size=size, start=first, end=last)
            )
    if not selected:
        raise CedaError(f"no files in {directory} overlap {start} to {end}")
    return sorted(selected, key=lambda item: item.start)


def total_bytes(files: list[ArchiveFile]) -> int:
    return sum(item.size for item in files)


def wget_command(url: str, destination: str = "data/raw/weather") -> str:
    """Return the CEDA bulk download command for everything below a URL."""
    return (
        'wget -e robots=off --mirror --no-parent -r '
        f'--header "Authorization: Bearer $CEDA_ACCESS_TOKEN" '
        f'-P {destination} {url}'
    )


def download(archive_file: ArchiveFile, destination: Path) -> Path:
    """Download one file. Raises if the archive returns a login page."""
    if not token():
        raise CedaAuthError(
            "CEDA_ACCESS_TOKEN is not set. Create a token at "
            "https://services.ceda.ac.uk/account/token/"
        )

    destination.mkdir(parents=True, exist_ok=True)
    path = destination / archive_file.name
    partial = destination / (archive_file.name + PARTIAL_SUFFIX)
    response = _get(archive_file.url, stream=True, allow_redirects=True)

    if response.status_code in AUTH_STATUS:
        raise CedaAuthError(
            f"{archive_file.name}: HTTP {response.status_code}, the token was refused"
        )
    if response.status_code != 200:
        raise CedaError(f"{archive_file.name}: HTTP {response.status_code}")
    if "html" in response.headers.get("content-type", ""):
        raise CedaAuthError(
            f"{archive_file.name}: the archive returned HTML rather than NetCDF, "
            f"which means the request was not authenticated"
        )

    received = 0
    with partial.open("wb") as handle:
        for chunk in response.iter_content(CHUNK):
            handle.write(chunk)
            received += len(chunk)

    if received != archive_file.size:
        partial.unlink(missing_ok=True)
        raise CedaError(
            f"{archive_file.name}: received {received} bytes, "
            f"listing says {archive_file.size}"
        )
    partial.replace(path)
    return path


@dataclass(frozen=True)
class Planned:
    """One listed file, where it belongs on disk and what is there already."""

    variable: str
    file: ArchiveFile
    directory: Path
    state: str

    @property
    def path(self) -> Path:
        return self.directory / self.file.name


@dataclass
class DownloadRun:
    """Running totals of a bulk download, kept current after every file."""

    files_expected: int = 0
    bytes_expected: int = 0
    files_present: int = 0
    bytes_present: int = 0
    files_skipped: int = 0
    files_downloaded: int = 0
    bytes_downloaded: int = 0
    failures: list[dict] = field(default_factory=list)
    stopped: str | None = None


def collect(
    base_url: str,
    variables: list[str],
    version: str,
    start: date,
    end: date,
    limit: int | None = None,
    lister: Callable[..., list[ArchiveFile]] = daily_files,
) -> dict[str, list[ArchiveFile]]:
    """Return the listed monthly files per variable, the first `limit` of each if set."""
    listed = {}
    for variable in variables:
        files = lister(base_url, variable, version, start, end)
        listed[variable] = files[:limit] if limit is not None else files
    return listed


def plan_downloads(files: dict[str, list[ArchiveFile]], destination: Path) -> list[Planned]:
    """Return every listed file with its directory and its state on disk.

    Each variable has its own directory under `destination`. The state is
    present where the file is on disk at the listed size, wrong size where it
    is on disk at another size, and absent otherwise.
    """
    planned = []
    for variable, items in files.items():
        directory = destination / variable
        for item in items:
            path = directory / item.name
            if not path.exists():
                state = "absent"
            elif path.stat().st_size == item.size:
                state = "present"
            else:
                state = "wrong size"
            planned.append(Planned(variable, item, directory, state))
    return planned


def run_downloads(
    planned: list[Planned],
    run: DownloadRun | None = None,
    downloader: Callable[[ArchiveFile, Path], Path] = download,
    retries: int = RETRIES,
    pause: float = PAUSE_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    report: Callable[[str], None] | None = None,
) -> DownloadRun:
    """Fetch the files not already on disk at the listed size, in order.

    A file at the wrong size is removed before it is fetched again. A file
    that fails is tried `retries` more times with a pause between, then
    recorded in `failures` and the run moves on. An authentication failure is
    not retried, since the same token is refused again. After
    AUTH_FAILURE_LIMIT consecutive files are refused the run ends with
    `stopped` set.

    `run` is updated after every file, so the totals survive an interruption.
    """
    run = run if run is not None else DownloadRun()
    run.files_expected = len(planned)
    run.bytes_expected = sum(item.file.size for item in planned)
    say = report or (lambda line: None)
    refused = 0

    for number, item in enumerate(planned, start=1):
        size = item.file.size
        if item.state == "present":
            run.files_skipped += 1
            run.files_present += 1
            run.bytes_present += size
            continue
        if item.state == "wrong size":
            found = item.path.stat().st_size
            item.path.unlink()
            say(f"  {item.file.name}: on disk at {found:,} bytes, listed at {size:,}; removed")

        reason = None
        auth = False
        attempts = 0
        for attempt in range(1, retries + 2):
            attempts = attempt
            try:
                downloader(item.file, item.directory)
                reason = None
                break
            except CedaAuthError as error:
                reason, auth = str(error), True
                break
            except (CedaError, requests.RequestException, OSError) as error:
                reason, auth = f"{type(error).__name__}: {error}", False
                if attempt <= retries:
                    say(f"  {item.file.name}: attempt {attempt} failed ({reason}); retrying")
                    sleep(pause)

        if reason is None:
            refused = 0
            run.files_downloaded += 1
            run.bytes_downloaded += size
            run.files_present += 1
            run.bytes_present += size
            state = "downloaded"
        else:
            refused = refused + 1 if auth else 0
            run.failures.append(
                {
                    "variable": item.variable,
                    "file": item.file.name,
                    "reason": reason,
                    "attempts": attempts,
                    "authentication": auth,
                }
            )
            state = f"FAILED after {attempts} attempt{'s' if attempts > 1 else ''}: {reason}"

        done = run.bytes_present / 1e9
        left = (run.bytes_expected - run.bytes_present) / 1e9
        say(
            f"file {number} of {len(planned)}  {item.file.name}  {state}  "
            f"{done:.1f} GB done, {left:.1f} GB left"
        )

        if refused >= AUTH_FAILURE_LIMIT:
            run.stopped = (
                f"{AUTH_FAILURE_LIMIT} consecutive files were refused for "
                f"authentication, last at file {number} of {len(planned)}"
            )
            break
    return run

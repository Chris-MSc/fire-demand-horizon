"""Access to HadUK-Grid daily files on the CEDA archive.

Directory listings under dap.ceda.ac.uk are public and carry file sizes;
file downloads redirect to a login page unless a CEDA access token is sent as
a bearer header. Tokens come from https://services.ceda.ac.uk/account/token/.

Daily files hold one month each, named
<variable>_hadukgrid_uk_1km_day_YYYYMMDD-YYYYMMDD.nc. A listing entry that does
not match that pattern stops the run.
"""

from __future__ import annotations

import getpass
import os
import re
from dataclasses import dataclass
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


class CedaError(RuntimeError):
    """Raised when the archive does not respond as expected."""


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
        raise CedaError(
            "CEDA_ACCESS_TOKEN is not set. Create a token at "
            "https://services.ceda.ac.uk/account/token/"
        )

    destination.mkdir(parents=True, exist_ok=True)
    path = destination / archive_file.name
    response = _get(archive_file.url, stream=True, allow_redirects=True)

    if response.status_code != 200:
        raise CedaError(f"{archive_file.name}: HTTP {response.status_code}")
    if "html" in response.headers.get("content-type", ""):
        raise CedaError(
            f"{archive_file.name}: the archive returned HTML rather than NetCDF, "
            f"which means the request was not authenticated"
        )

    received = 0
    with path.open("wb") as handle:
        for chunk in response.iter_content(CHUNK):
            handle.write(chunk)
            received += len(chunk)

    if received != archive_file.size:
        raise CedaError(
            f"{archive_file.name}: received {received} bytes, "
            f"listing says {archive_file.size}"
        )
    return path

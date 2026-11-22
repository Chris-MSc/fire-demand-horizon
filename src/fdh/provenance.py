"""File hashes for the build records.

A built artefact records the hash of each input it was built from, so a later
run can tell an artefact built from other inputs from one that is merely
incomplete.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

CHUNK = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()

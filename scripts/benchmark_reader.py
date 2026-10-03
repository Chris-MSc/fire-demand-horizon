"""Compare pandas with odfpy against the streaming reader on one sheet.

Reports the row count and elapsed time for each reader. The row counts agree
only if pandas expands table:number-rows-repeated.

pandas builds the whole document in memory. The streaming reader emits rows as they are parsed, 
so it can handle a 363 MB content.xml without running out of memory. The streaming reader is therefore the only option for 
the 2023-24 to present file, which is 1.8 million rows and 1.2 GB uncompressed.

Usage:
    uv run python scripts/benchmark_reader.py --era 2023-24_to_present --sheet 202324
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdh.config import load_config  # noqa: E402
from fdh.ods import iter_rows  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--era", required=True)
    parser.add_argument("--sheet", required=True)
    parser.add_argument("--skip-pandas", action="store_true")
    args = parser.parse_args()

    config = load_config()
    raw_dir = config.paths.resolved("raw_dir")
    source = next(
        (
            item
            for sources in config.sources.values()
            for item in sources
            if item.era == args.era
        ),
        None,
    )
    if source is None:
        raise SystemExit(f"no source in config with era {args.era}")
    path = raw_dir / source.filename
    print(f"{path.name}, sheet {args.sheet}\n")

    start = time.perf_counter()
    streaming_rows = sum(1 for _ in iter_rows(path, sheet=args.sheet))
    streaming_seconds = time.perf_counter() - start
    print(f"streaming: {streaming_rows:,} rows including header, {streaming_seconds:.1f} s")

    if args.skip_pandas:
        return 0

    import pandas as pd

    print("pandas:    running. Watch resident memory.")
    start = time.perf_counter()
    try:
        frame = pd.read_excel(path, sheet_name=args.sheet, engine="odf", dtype=str)
    except MemoryError:
        print("pandas:    MemoryError. Use the streaming reader.")
        return 0
    pandas_seconds = time.perf_counter() - start
    pandas_rows = len(frame) + 1  # plus header, which pandas consumes
    print(f"pandas:    {pandas_rows:,} rows including header, {pandas_seconds:.1f} s")

    print()
    if pandas_rows == streaming_rows:
        print("Row counts agree: pandas honours the row repetition attribute.")
    else:
        difference = streaming_rows - pandas_rows
        print(
            f"Row counts DISAGREE by {difference:,} rows "
            f"({difference / streaming_rows:.1%} of the streaming count)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
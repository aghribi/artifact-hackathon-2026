"""Per-pulse features of a DD parquet taken out of the capture tars, such as one filtered to a single cavity.

The input has the rows and columns of a DD parquet: one row per pv and cycle, the waveform in data_array. The
output has the rows and columns of cavity_pulses.parquet of ess_dd_features.py, computed by the same functions,
without the beam and trip columns: those need the BCM waveforms and the Archiver, which a filtered file does not hold.

    python ess_dd_subset_features.py <dd.parquet> <out.parquet> <n_bins>
"""

import sys
from pathlib import Path

import pandas as pd

from ess_dd_features import add_cycle_context, parquet_rows


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    source, out, n_bins = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    # The file is not inside a capture folder, so its own name stands in for the capture window.
    cavity_rows, _ = parquet_rows(source, source.stem, n_bins)
    df = add_cycle_context(pd.DataFrame(cavity_rows))
    df.to_parquet(out, index=False)
    print(f"{len(df)} pulses, {df.cavity.nunique()} cavities, {df.event_id.nunique()} captures, "
          f"{int((~df.has_pulse).sum())} without pulse -> {out}")


if __name__ == "__main__":
    main()

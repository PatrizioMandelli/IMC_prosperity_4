import glob
from pathlib import Path

import pandas as pd


class RoundRegistry:
    """Auto-discovers available rounds and products from a data/ directory."""

    @staticmethod
    def scan(data_root="data") -> dict[int, list[int]]:
        """
        Returns {round_n: [day_int, ...]} by globbing data/roundN/prices_round_N_day_D.csv.
        """
        data_root = Path(data_root)
        result: dict[int, list[int]] = {}

        for f in sorted(data_root.glob("round*/prices_round_*_day_*.csv")):
            stem = f.stem  # e.g. "prices_round_2_day_-1"
            parts = stem.split("_")
            try:
                round_n = int(parts[2])
                day = int(parts[4])
            except (IndexError, ValueError):
                continue
            result.setdefault(round_n, [])
            if day not in result[round_n]:
                result[round_n].append(day)

        return {k: sorted(v) for k, v in sorted(result.items())}

    @staticmethod
    def products_for(data_root, round_n: int) -> list[str]:
        """Returns unique product names found in the first available price file for that round."""
        data_root = Path(data_root)
        files = sorted((data_root / f"round{round_n}").glob(f"prices_round_{round_n}_day_*.csv"))
        if not files:
            return []
        df = pd.read_csv(files[0], sep=";", on_bad_lines="skip", engine="python", nrows=500)
        df.columns = df.columns.str.strip()
        col = "product" if "product" in df.columns else df.columns[2]
        return sorted(df[col].dropna().unique().tolist())

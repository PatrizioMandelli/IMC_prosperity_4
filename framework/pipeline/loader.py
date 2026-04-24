import csv
import io
import json
import re
import sys
from pathlib import Path

import pandas as pd

TICKS_PER_DAY = 1_000_000

# Column list shared with normalizer
PRICE_NUMS = [
    "day", "timestamp",
    "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2", "bid_price_3", "bid_volume_3",
    "ask_price_1", "ask_volume_1", "ask_price_2", "ask_volume_2", "ask_price_3", "ask_volume_3",
    "mid_price", "profit_and_loss",
]


class UniversalLoader:
    """
    Single entry point for all data sources.
    Returns (prices_df, trades_df) with a monotonic global_time column.
    """

    @staticmethod
    def from_content(content: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Parses backtest log content directly from a string.
        """
        activities_text = None
        trades_text = None

        # 1. Try to parse as official IMC JSON
        if content.strip().startswith("{") and content.strip().endswith("}"):
            try:
                data = json.loads(content)
                raw = data.get("activitiesLog", "") or data.get("sandboxLogs", "") or data.get("logs", "")
                if not raw and "activities" in data:
                    raw = data["activities"]
                
                if isinstance(raw, list):
                    raw = "\n".join(str(l) for l in raw)
                
                if raw:
                    # Robustly handle the newline escaping in some JSON logs
                    raw_cleaned = raw.replace("\\n", "\n")
                    if "Activities log:" not in raw_cleaned:
                        content = "Activities log:\n" + raw_cleaned
                    else:
                        content = raw_cleaned
                        
                    if "tradeHistory" in data:
                        trades_text = json.dumps(data["tradeHistory"])
            except json.JSONDecodeError:
                pass

        # 2. Extract sections
        act_match = re.search(r"Activities\s*log:?\s*", content, re.IGNORECASE)
        if act_match:
            parts = content[act_match.end():].split("Trade History:")
            activities_text = parts[0].strip()
            if len(parts) > 1 and not trades_text:
                trades_text = parts[1].strip()
        else:
            if ";" in content and ("day" in content.lower() or "timestamp" in content.lower()):
                activities_text = content.strip()

        prices_df = pd.DataFrame()
        if activities_text:
            try:
                clean_lines = []
                for line in activities_text.splitlines():
                    line = line.strip()
                    if not line: continue
                    line = re.sub(r"^\[.*?\]\s*", "", line)
                    clean_lines.append(line)
                
                activities_text = "\n".join(clean_lines)
                prices_df = pd.read_csv(
                    io.StringIO(activities_text), sep=";",
                    on_bad_lines="skip", engine="python",
                )
                prices_df.columns = prices_df.columns.str.strip()
                if not prices_df.empty and "day" in prices_df.columns:
                    prices_df = prices_df[pd.to_numeric(prices_df["day"], errors="coerce").notna()]
                
                for c in PRICE_NUMS:
                    if c in prices_df.columns:
                        prices_df[c] = pd.to_numeric(prices_df[c], errors="coerce")
                
                if not prices_df.empty:
                    prices_df = prices_df.sort_values(["day", "timestamp"]).reset_index(drop=True)
                    prices_df = UniversalLoader._add_global_time(prices_df)
            except Exception as e:
                print(f"Error parsing activities: {e}")

        trades_list = []
        if trades_text:
            try:
                trades_list = json.loads(trades_text)
            except json.JSONDecodeError:
                for match in re.finditer(r"\{[^{}]+\}", trades_text):
                    block = match.group(0)
                    try:
                        ts_m = re.search(r'"timestamp":\s*(\d+)', block)
                        buyer_m = re.search(r'"buyer":\s*"([^"]*)"', block)
                        seller_m = re.search(r'"seller":\s*"([^"]*)"', block)
                        symbol_m = re.search(r'"symbol":\s*"([^"]*)"', block)
                        qty_m = re.search(r'"quantity":\s*(\d+)', block)
                        price_m = re.search(r'"price":\s*([\d.]+)', block)
                        if ts_m and symbol_m and qty_m:
                            trades_list.append({
                                "timestamp": int(ts_m.group(1)),
                                "buyer": buyer_m.group(1) if buyer_m else "",
                                "seller": seller_m.group(1) if seller_m else "",
                                "symbol": symbol_m.group(1),
                                "quantity": int(qty_m.group(1)),
                                "price": float(price_m.group(1)) if price_m else float("nan"),
                            })
                    except Exception:
                        pass

        trades_df = pd.DataFrame(trades_list)
        if not trades_df.empty and "timestamp" in trades_df.columns:
            trades_df = trades_df.sort_values("timestamp").reset_index(drop=True)
            if "day" not in trades_df.columns:
                trades_df["day"] = 0
            trades_df = UniversalLoader._add_global_time(trades_df)
            if "symbol" in trades_df.columns and "product" not in trades_df.columns:
                trades_df = trades_df.rename(columns={"symbol": "product"})

        return prices_df, trades_df

    @staticmethod
    def from_json_log(path) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads a backtest .log / .json file from a path.
        """
        path = Path(path)
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = path.read_text(encoding="latin-1")
        return UniversalLoader.from_content(content)

    @staticmethod
    def from_csv_dir(
        data_dir,
        round_n: int,
        days: list[int] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads price+trade CSVs from data/roundN/ for the given round and days.
        If days=None, auto-discovers all available day files.
        """
        data_dir = Path(data_dir)
        round_dir = data_dir / f"round{round_n}"

        if days is None:
            import glob as _glob
            found = _glob.glob(str(round_dir / f"prices_round_{round_n}_day_*.csv"))
            days = sorted({
                int(Path(f).stem.split("_day_")[-1])
                for f in found
            })

        days_sorted = sorted(days)
        prices_list, trades_list = [], []

        for rank, day in enumerate(days_sorted):
            pf = round_dir / f"prices_round_{round_n}_day_{day}.csv"
            tf = round_dir / f"trades_round_{round_n}_day_{day}.csv"

            if pf.exists():
                df_p = pd.read_csv(pf, sep=";", on_bad_lines="skip", engine="python")
                df_p.columns = df_p.columns.str.strip()
                for c in PRICE_NUMS:
                    if c in df_p.columns:
                        df_p[c] = pd.to_numeric(df_p[c], errors="coerce")
                df_p["day"] = day
                df_p["_day_rank"] = rank
                df_p["global_time"] = rank * TICKS_PER_DAY + df_p["timestamp"]
                prices_list.append(df_p)

            if tf.exists():
                df_t = pd.read_csv(tf, sep=";", on_bad_lines="skip", engine="python")
                df_t.columns = df_t.columns.str.strip()
                df_t["day"] = day
                df_t["_day_rank"] = rank
                df_t["global_time"] = rank * TICKS_PER_DAY + df_t["timestamp"]
                if "symbol" in df_t.columns and "product" not in df_t.columns:
                    df_t = df_t.rename(columns={"symbol": "product"})
                trades_list.append(df_t)

        prices_df = (
            pd.concat(prices_list, ignore_index=True)
            .sort_values(["day", "timestamp"])
            .reset_index(drop=True)
            if prices_list else pd.DataFrame()
        )
        trades_df = (
            pd.concat(trades_list, ignore_index=True)
            .sort_values(["day", "timestamp"])
            .reset_index(drop=True)
            if trades_list else pd.DataFrame()
        )

        return prices_df, trades_df

    @staticmethod
    def _add_global_time(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds global_time = day_rank * TICKS_PER_DAY + timestamp.
        day_rank is the sort-order index of unique day values (handles negatives like -2,-1,0).
        """
        if "global_time" in df.columns:
            return df
        days_sorted = sorted(df["day"].dropna().unique().astype(int))
        rank_map = {d: i for i, d in enumerate(days_sorted)}
        df = df.copy()
        df["global_time"] = df["day"].map(rank_map) * TICKS_PER_DAY + df["timestamp"]
        return df

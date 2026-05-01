import numpy as np
import pandas as pd

ROLL_W = 30


def _oba_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    median = df["mid_price"].median()
    df.loc[df["mid_price"] < median * 0.5, "mid_price"] = np.nan
    df["mid_price"] = df["mid_price"].interpolate()

    df["bid_depth"] = df[["bid_volume_1", "bid_volume_2", "bid_volume_3"]].sum(axis=1, min_count=1)
    df["ask_depth"] = df[["ask_volume_1", "ask_volume_2", "ask_volume_3"]].sum(axis=1, min_count=1)
    df["one_sided"] = df["bid_depth"].eq(0) | df["ask_depth"].eq(0)
    df["spread"] = df["ask_price_1"] - df["bid_price_1"]
    df["spread_roll"] = df["spread"].rolling(ROLL_W, min_periods=1).mean()
    df["pnl_delta"] = df["profit_and_loss"].diff().fillna(0) if "profit_and_loss" in df.columns else 0
    return df


class LOBNormalizer:
    """Enriches raw price DataFrames with micro_price, OFI, and trade classification."""

    @staticmethod
    def normalize(prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Per-product enrichment:
          1. Runs order_book_analyzer.features() (outlier removal, depth, spread, pnl_delta)
          2. Adds micro_price (WallMid) from level-1 bid/ask
          3. Adds ofi (Order Flow Imbalance, normalized)
        """
        if prices_df.empty:
            return prices_df

        parts = []
        for _, grp in prices_df.groupby("product", sort=False):
            grp = grp.sort_values("global_time").reset_index(drop=True)
            grp = _oba_features(grp)
            grp["micro_price"] = LOBNormalizer._micro_price_col(grp)
            grp["ofi"] = LOBNormalizer._ofi_col(grp)
            parts.append(grp)

        out = pd.concat(parts, ignore_index=True).sort_values("global_time").reset_index(drop=True)
        return out

    @staticmethod
    def classify_trades(trades_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Joins trades to the nearest price snapshot and classifies each trade as MAKER or TAKER.
        Adds columns: maker_taker, mid_at_trade, dist_from_mid.
        """
        if trades_df.empty or prices_df.empty:
            return trades_df

        price_ref = prices_df[
            ["global_time", "product", "mid_price", "bid_price_1", "ask_price_1"]
        ].copy().sort_values("global_time")

        trades_sorted = trades_df.sort_values("global_time").copy()

        result_parts = []
        products = trades_sorted["product"].dropna().unique()
        for prod in products:
            t = trades_sorted[trades_sorted["product"] == prod].copy()
            p = price_ref[price_ref["product"] == prod]
            if p.empty:
                t["maker_taker"] = "UNKNOWN"
                t["mid_at_trade"] = float("nan")
                t["dist_from_mid"] = float("nan")
                result_parts.append(t)
                continue

            merged = pd.merge_asof(
                t,
                p[["global_time", "mid_price", "bid_price_1", "ask_price_1"]],
                on="global_time",
                direction="backward",
            )
            merged["mid_at_trade"] = merged["mid_price"]
            merged["dist_from_mid"] = merged["price"] - merged["mid_price"]

            def _classify(row):
                price = row.get("price", float("nan"))
                bid1 = row.get("bid_price_1", float("nan"))
                ask1 = row.get("ask_price_1", float("nan"))
                if pd.isna(price):
                    return "UNKNOWN"
                if not pd.isna(ask1) and price >= ask1:
                    return "TAKER"   # aggressive buy: lifted the ask
                if not pd.isna(bid1) and price <= bid1:
                    return "TAKER"   # aggressive sell: hit the bid
                return "MAKER"       # trade inside spread → passive/maker

            merged["maker_taker"] = merged.apply(_classify, axis=1)
            merged = merged.drop(columns=["mid_price", "bid_price_1", "ask_price_1"], errors="ignore")
            result_parts.append(merged)

        return pd.concat(result_parts, ignore_index=True).sort_values("global_time").reset_index(drop=True)

    @staticmethod
    def _micro_price_col(df: pd.DataFrame) -> pd.Series:
        """
        WallMid = (bid_vol_1 * ask_price_1 + ask_vol_1 * bid_price_1) / (bid_vol_1 + ask_vol_1)
        Returns NaN when either side has zero volume; forward-filled up to 3 ticks.
        """
        bv = df["bid_volume_1"].fillna(0)
        av = df["ask_volume_1"].fillna(0)
        bp = df["bid_price_1"]
        ap = df["ask_price_1"]

        total = bv + av
        mp = np.where(total > 0, (bv * ap + av * bp) / total, np.nan)
        result = pd.Series(mp, index=df.index).ffill()
        if "mid_price" in df.columns:
            result = result.fillna(df["mid_price"])
        return result

    @staticmethod
    def _ofi_col(df: pd.DataFrame) -> pd.Series:
        """
        OFI = (Δbid_vol_1 - Δask_vol_1) / max(bid_depth + ask_depth, 1)
        First row is 0.
        """
        bv = df["bid_volume_1"].fillna(0)
        av = df["ask_volume_1"].fillna(0)
        total_depth = (
            df.get("bid_depth", bv) + df.get("ask_depth", av)
        ).clip(lower=1)

        delta_bid = bv.diff().fillna(0)
        delta_ask = av.diff().fillna(0)
        return ((delta_bid - delta_ask) / total_depth).fillna(0)

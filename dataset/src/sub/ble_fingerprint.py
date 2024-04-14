from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.realpath(__file__)))


# ble_fingerのデータ構造
# ts,x,y,z,beacon_address,rssi,floor_name
def round_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Coordinates rounding to one decimal place."""
    for coord in ["x", "y", "z"]:
        df[coord] = df[coord].apply(lambda x: round(x, 1))
    return df


def aggregate_beacon_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate RSSI statistics for each beacon location."""
    agg_df = (
        df.groupby(["x", "y", "z", "beacon_address", "floor"])
        .agg(
            rssi_mean=pd.NamedAgg("rssi", "mean"),
            rssi_std=pd.NamedAgg("rssi", "std"),
            count=pd.NamedAgg("rssi", "count"),
        )
        .reset_index()
    )
    agg_df["rssi_std"] = agg_df["rssi_std"].fillna(0)  # Replace NaN variance with 0
    return agg_df


def filter_beacons_by_floor_and_rssi(
    df: pd.DataFrame,
    floor_name: str,
    rssi_threshold: int,
) -> np.ndarray:
    """Filter beacons by floor and a minimum RSSI threshold."""
    filtered_df = df[(df["floor"] == floor_name) & (df["rssi_mean"] > rssi_threshold)]
    return filtered_df["beacon_address"].unique()


def process_beacon_data(
    df: pd.DataFrame,
    beacon_addresses: np.ndarray,
    floor_name: str,
    rssi_threshold: int,
) -> list[pd.DataFrame]:
    """Further processing of beacon data for given addresses on a specific floor."""
    all_results = []
    for address in beacon_addresses:
        subset = df[
            (df["beacon_address"] == address)
            & (df["floor"] == floor_name)
            & (df["rssi_mean"] > rssi_threshold)
        ]
        count_threshold = np.average(subset["count"], weights=subset["count"])
        refined_df = subset[subset["count"] >= count_threshold].reset_index(drop=True)
        refined_df["third_quartile"] = subset["count"].quantile(0.75)
        refined_df["count_mean"] = subset["count"].mean()
        refined_df["count_weight_average"] = count_threshold
        all_results.append(refined_df)
    return all_results


def ble_finger(ble_fingerprint_df: pd.DataFrame):
    """Process BLE fingerprint data."""
    ble_fingerprint_df = round_coordinates(ble_fingerprint_df)
    beacon_stats_df = aggregate_beacon_data(ble_fingerprint_df)
    beacon_addresses = filter_beacons_by_floor_and_rssi(beacon_stats_df, "FLU01", -70)
    processed_beacon_data = process_beacon_data(
        beacon_stats_df,
        beacon_addresses,
        "FLU01",
        -70,
    )
    return processed_beacon_data


def main():
    ble_fingerprint_df = pd.read_csv("../beacon_reception_events.csv")
    processed_beacon_data = ble_finger(ble_fingerprint_df)
    print(processed_beacon_data[0])
    print(processed_beacon_data[1])
    print(processed_beacon_data[2])
    print(processed_beacon_data[3])
    print(processed_beacon_data[4])
    print(processed_beacon_data[4])


if __name__ == "__main__":
    main()

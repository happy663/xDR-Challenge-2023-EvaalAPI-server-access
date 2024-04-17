from __future__ import annotations

import os
import sys
from typing import Literal

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import ble
import estimate


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
) -> pd.DataFrame:
    """Further processing of beacon data for given addresses on a specific floor."""
    all_results_df = pd.DataFrame()
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

        all_results_df = pd.concat([all_results_df, refined_df], ignore_index=True)

    return all_results_df


def aggregate_consecutive_bdaddress(df):
    """Aggregate consecutive rows with the same bdaddress by averaging their ts and rssi values, keeping the bdaddress in the final output."""
    # Identify changes in bdaddress to define groups
    df["group"] = (df["bdaddress"] != df["bdaddress"].shift()).cumsum()

    # Calculate mean ts and rssi for each group and include bdaddress in the results
    aggregated_df = (
        df.groupby(["bdaddress", "group"])
        .agg(
            {
                "ts": "mean",
                "rssi": "mean",
            },
        )
        .reset_index()
        .drop(columns=["group"])
    )  # Removing the 'group' column after reset_index

    return aggregated_df


Axis2D = Literal["x", "y"]


def rotate_trajectory_to_optimal_alignment_using_ble_fingerprint(
    acc_df: pd.DataFrame,
    angle_df: pd.DataFrame,
    ble_scans_df: pd.DataFrame,
    ble_fingerprint_df: pd.DataFrame,
    floor_name: str,
    *,
    ground_truth_first_point: dict[Axis2D, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rotate the trajectory to the optimal alignment using BLE fingerprint data.

    Args:
    ----
        acc_df (pd.DataFrame): Accelerometer data.
        angle_df (pd.DataFrame): Angle data.
        ble_scans_df (pd.DataFrame): BLE scans data.
        ble_fingerprint_df (pd.DataFrame): BLE fingerprint data.
        floor_name (str): Name of the floor.
        ground_truth_first_point (Optional[Dict[Axis2D, float]], optional): Ground truth first point. Defaults to None.

    Returns:
    -------
        tuple[pd.DataFrame, pd.DataFrame]: Tuple containing the rotated optimal angle and displacement data.

    """
    if ground_truth_first_point is None:
        ground_truth_first_point = {"x": 0.0, "y": 0.0}
    first_time_remove_drift_angle_displacement = (
        estimate.convert_to_peek_angle_and_compute_displacement_by_angle(
            angle_df,
            acc_df,
            0.5,
            {"x": ground_truth_first_point["x"], "y": ground_truth_first_point["y"]},
            0.0,
        )
    )

    strong_ble_scans_df = estimate.filter_strong_blescans(ble_scans_df, 180, -76)

    """Process BLE fingerprint data."""
    aggreated_strong_blescans = (
        aggregate_consecutive_bdaddress(
            strong_ble_scans_df,
        )
        .sort_values("ts")
        .reset_index(drop=True)
    )
    ble_fingerprint_df = round_coordinates(ble_fingerprint_df)
    aggregate_ble_fingerprint_df = aggregate_beacon_data(ble_fingerprint_df)
    beacon_addresses_array = filter_beacons_by_floor_and_rssi(
        aggregate_ble_fingerprint_df,
        "FLU01",
        -70,
    )

    beacon_stats_df = process_beacon_data(
        aggregate_ble_fingerprint_df,
        beacon_addresses_array,
        floor_name,
        -70,
    )

    merged_beacon_stats_df = beacon_stats_df.merge(
        aggreated_strong_blescans,
        left_on="beacon_address",
        right_on="bdaddress",
        how="left",
    )

    delete_nan_merged_beacon_stats_df = (
        merged_beacon_stats_df.dropna(subset=["bdaddress"])
        .sort_values("ts")
        .reset_index(drop=True)
    )

    print(
        delete_nan_merged_beacon_stats_df.to_csv(
            "delete_nan_merged_beacon_stats_df.csv"
        )
    )

    optimal_angle = ble.search_optimal_angle(
        first_time_remove_drift_angle_displacement,
        delete_nan_merged_beacon_stats_df,
        ground_truth_first_point={
            "x": ground_truth_first_point["x"],
            "y": ground_truth_first_point["y"],
        },
    )
    print(optimal_angle)

    rotated_optimal_angle_df = pd.DataFrame(
        {
            "ts": angle_df["ts"],
            "x": angle_df["x"] + optimal_angle,
        },
    )

    rotated_optimal_angle_df_displacement = (
        estimate.convert_to_peek_angle_and_compute_displacement_by_angle(
            rotated_optimal_angle_df,
            acc_df,
            0.5,
            {"x": ground_truth_first_point["x"], "y": ground_truth_first_point["y"]},
        )
    )

    return (
        rotated_optimal_angle_df,
        rotated_optimal_angle_df_displacement,
        delete_nan_merged_beacon_stats_df,
    )


def main() -> None:
    """Entry point of the program."""
    ble_fingerprint_df = pd.read_csv("../beacon_reception_events.csv")


if __name__ == "__main__":
    main()

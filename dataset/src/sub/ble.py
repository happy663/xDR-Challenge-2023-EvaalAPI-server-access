from typing import Literal

import estimate
import numpy as np
import pandas as pd

Axis2D = Literal["x", "y"]


def rotate_cumulative_displacement(df, angle, initial_point: dict[str, float]):
    x_displacement = df.x_displacement.values - initial_point["x"]
    y_displacement = df.y_displacement.values - initial_point["y"]

    # rotate displacement using numpy
    x_new = x_displacement * np.cos(angle) - y_displacement * np.sin(angle)
    y_new = x_displacement * np.sin(angle) + y_displacement * np.cos(angle)

    # 元の位置に戻す
    x_new += initial_point["x"]
    y_new += initial_point["y"]

    return pd.DataFrame({"ts": df.ts, "x_displacement": x_new, "y_displacement": y_new})


def search_optimal_angle(
    displacement_df: pd.DataFrame,
    strong_ble_merege_df: pd.DataFrame,
    *,
    ground_truth_first_point: dict[Axis2D, float],
) -> float:
    # 探索する角度の範囲
    angle_range = np.arange(0, 2 * np.pi, 0.01)
    angle_and_euclidean_list: list[dict[str, float]] = []

    for angle in angle_range:
        new_df = rotate_cumulative_displacement(
            displacement_df,
            angle,
            {"x": ground_truth_first_point["x"], "y": ground_truth_first_point["y"]},
        )

        # Find nearest rows using merge_asof
        merged_df = pd.merge_asof(
            new_df.sort_values("ts"),
            strong_ble_merege_df.sort_values("ts"),
            on="ts",
            direction="nearest",
        )

        # Calculate Euclidean distances
        merged_df["euclidean_distance"] = np.sqrt(
            (merged_df["x_displacement"] - merged_df["x"]) ** 2
            + (merged_df["y_displacement"] - merged_df["y"]) ** 2,
        )

        total_euclidean_distance = merged_df["euclidean_distance"].sum()

        angle_and_euclidean_list.append(
            {"angle": angle, "total_euclidean_distance": total_euclidean_distance},
        )

    # 一番ユークリッド距離が小さいangleとeuclidean_distanceを取得
    angle_and_euclidean = min(
        angle_and_euclidean_list,
        key=lambda x: x["total_euclidean_distance"],
    )

    return angle_and_euclidean["angle"]


def rotate_trajectory_to_optimal_alignment_using_ble(
    acc_df: pd.DataFrame,
    angle_df: pd.DataFrame,
    ble_scans_df: pd.DataFrame,
    ble_position_df: pd.DataFrame,
    *,
    ground_truth_first_point: dict[Axis2D, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if ground_truth_first_point is None:
        ground_truth_first_point = {"x": 0.0, "y": 0.0}

    cumulative_displacement_df = (
        estimate.convert_to_peek_angle_and_compute_displacement_by_angle(
            angle_df,
            acc_df,
            0.5,
            {"x": ground_truth_first_point["x"], "y": ground_truth_first_point["y"]},
        )
    )

    strong_ble_scans_df = estimate.filter_strong_blescans(
        ble_scans_df,
        90,
        -76,
    )

    strong_ble_merege_df = strong_ble_scans_df.merge(
        ble_position_df,
        on="bdaddress",
        how="left",
    )
    print(strong_ble_merege_df)

    optimal_angle = search_optimal_angle(
        cumulative_displacement_df,
        strong_ble_merege_df,
        ground_truth_first_point=ground_truth_first_point,
    )

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

    return rotated_optimal_angle_df, rotated_optimal_angle_df_displacement

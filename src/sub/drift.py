from __future__ import annotations

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from typing import TYPE_CHECKING

import estimate

if TYPE_CHECKING:
    import pandas as pd


def _apply_remove_drift_to_angle(
    angle: pd.DataFrame,
    drift: float,
    base_time: pd.Timestamp,
) -> pd.DataFrame:
    """角度にドリフト除去を適用する."""
    angle_with_drift = angle.copy()
    elapsed_time = angle_with_drift["ts"] - base_time
    angle_with_drift["x"] -= drift * elapsed_time
    return angle_with_drift


def _compute_euclidean_distance(df: pd.DataFrame, gt: pd.Series) -> float:
    """軌跡の最後の座標と正解座標のユークリッド距離を計算する."""
    last_row = df.iloc[-1]
    return np.sqrt(
        (last_row["x_displacement"] - gt.x) ** 2
        + (last_row["y_displacement"] - gt.y) ** 2,
    )


def _search_optimal_drift_from_angle(
    acc_df: pd.DataFrame,
    angle_df: pd.DataFrame,
    gt_ref: pd.DataFrame,
) -> dict:
    """角度のデータからドリフトを探索する."""
    drift_range = np.arange(-0.05, 0.05, 0.001)
    base_time = angle_df["ts"].iloc[0]

    drift_and_euclidean_list = []
    for drift in drift_range:
        adjusted_angle = _apply_remove_drift_to_angle(angle_df, drift, base_time)
        displacement_df = (
            estimate.convert_to_peek_angle_and_compute_displacement_by_angle(
                adjusted_angle,
                acc_df,
                0.5,
                {"x": gt_ref.x[0], "y": gt_ref.y[0]},
                gt_ref["ts"][0],
            )
        )
        displacement_df = displacement_df.reset_index(drop=True)

        euclidean_distance = _compute_euclidean_distance(
            displacement_df,
            gt_ref.iloc[1],
        )
        drift_and_euclidean_list.append(
            {"drift": drift, "euclidean_distance": euclidean_distance},
        )

    # 絶対値が0.01以上のドリフトを除外
    drift_and_euclidean_list = [
        drift_and_euclidean
        for drift_and_euclidean in drift_and_euclidean_list
        if abs(drift_and_euclidean["drift"]) < 0.01
    ]

    sorted_drift_and_euclidean_list = sorted(
        drift_and_euclidean_list,
        key=lambda x: x["euclidean_distance"],
    )

    return min(
        sorted_drift_and_euclidean_list,
        key=lambda x: x["euclidean_distance"],
    )


def remove_drift_in_angle_df(
    acc_df: pd.DataFrame,
    angle_df: pd.DataFrame,
    ground_truth_point_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove drift from angle data.

    Args:
    ----
        acc_df (pd.DataFrame): Acceleration data.
        angle_df (pd.DataFrame): Angle data.
        ground_truth_point_df (pd.DataFrame): Ground tooth 2D coordinate data.

    Returns:
    -------
        tuple[pd.DataFrame, pd.DataFrame]: Tuple containing the angle data with drift removed and the corrected angle displacement data.

    """
    # ドリフトの探索
    optimal_drift_and_euclidean = _search_optimal_drift_from_angle(
        acc_df,
        angle_df,
        ground_truth_point_df,
    )

    # ドリフトの除去の適用
    remove_drift_angles_df = _apply_remove_drift_to_angle(
        angle_df,
        optimal_drift_and_euclidean["drift"],
        angle_df["ts"].iloc[0],
    )

    # 変位の計算
    corrected_angle_displacement = (
        estimate.convert_to_peek_angle_and_compute_displacement_by_angle(
            remove_drift_angles_df,
            acc_df,
            0.5,
            {
                "x": ground_truth_point_df.x[0],
                "y": ground_truth_point_df.y[0],
            },
            ground_truth_point_df["ts"][0],
        )
    )

    return remove_drift_angles_df, corrected_angle_displacement

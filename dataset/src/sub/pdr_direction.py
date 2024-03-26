from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# fmt: off
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import estimate
import utils

# fmt: on


def _calculate_horizontal_and_vertical_counts(
    angle_df: pd.DataFrame,
    rotate_angle: float,
) -> dict[str, int | float]:
    rotated_angle = (angle_df["x"] + rotate_angle) % (2 * np.pi)

    vertical_count = len(
        rotated_angle[
            ((rotated_angle >= np.pi / 2 - 0.1) & (rotated_angle <= np.pi / 2 + 0.1))
            | (
                (rotated_angle >= 3 * np.pi / 2 - 0.1)
                & (rotated_angle <= 3 * np.pi / 2 + 0.1)
            )
        ],
    )

    horizontal_count = len(
        rotated_angle[
            ((rotated_angle <= 0.1) | (rotated_angle >= 2 * np.pi - 0.1))
            | ((rotated_angle >= np.pi - 0.1) & (rotated_angle <= np.pi + 0.1))
        ],
    )

    return {
        "angle": rotate_angle,
        "horizontal_and_vertical_count": horizontal_count + vertical_count,
    }


def _calculate_exist_counts(
    angle_df: pd.DataFrame,
    results: pd.DataFrame,
    gt_ref: pd.DataFrame,
    edit_map_dict: dict[str, np.ndarray],
    floor_name: str,
    dx: float,
    dy: float,
) -> pd.DataFrame:
    for i, row in results.head(20).iterrows():
        rotated_displacement = estimate.calculate_cumulative_displacement(
            angle_df.ts,
            (angle_df["x"] + row["angle"]),
            0.5,
            {"x": gt_ref.x[0], "y": gt_ref.y[0]},
            gt_ref["%time"][0],
        )

        exist_count = 0
        for _, displacement_row in rotated_displacement.iterrows():
            if _is_passable(
                edit_map_dict,
                floor_name,
                displacement_row["x_displacement"],
                displacement_row["y_displacement"],
                dx,
                dy,
            ):
                exist_count += 1

        results.at[i, "exist_count"] = exist_count

    return results


def _get_optimal_angle(results: pd.DataFrame) -> float:
    max_exist_count = results["exist_count"].max()
    optimal_result = (
        results[results["exist_count"] == max_exist_count]
        .sort_values(by="horizontal_and_vertical_count", ascending=False)
        .iloc[0]
    )
    return optimal_result["angle"]


def _is_passable(
    passable_dict: dict[str, np.ndarray],
    floor_name: str,
    x: float,
    y: float,
    dx: float,
    dy: float,
) -> bool:
    epsilon = 1e-9  # 非常に小さい値
    # 不動小数点の切り捨てによる誤差を防ぐために、微小な値を足している
    # 例えば32.51の場合微小な値を足さないと3250.9999999999995となり、3250に切り捨てられてしまう
    row = int((x + epsilon) / dx)
    col = int((y + epsilon) / dy)

    #  numpy配列の範囲外にアクセスしようとした場合はFalseを返す
    if (
        row < 0
        or col < 0
        or row >= passable_dict[floor_name].shape[0]
        or col >= passable_dict[floor_name].shape[1]
    ):
        return False

    return passable_dict[floor_name][row, col]


def _find_best_alignment_angle(
    angle_df: pd.DataFrame,
    gt_ref: pd.DataFrame,
    edit_map_dict: dict[str, np.ndarray],
    floor_name: str,
    dx: float,
    dy: float,
) -> float:
    angle_range = np.arange(0, 2 * np.pi, 0.01)
    results = [
        _calculate_horizontal_and_vertical_counts(angle_df, rotate_angle)
        for rotate_angle in angle_range
    ]
    df_results = pd.DataFrame(results).sort_values(
        by="horizontal_and_vertical_count",
        ascending=False,
    )
    df_results = df_results.reset_index(drop=True)
    df_results = _calculate_exist_counts(
        angle_df,
        df_results,
        gt_ref,
        edit_map_dict,
        floor_name,
        dx,
        dy,
    )
    return _get_optimal_angle(df_results)


def find_best_direction(
    acc_df: pd.DataFrame,
    angle_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    edit_map_dict: dict[str, np.ndarray],
    floor_name: str,
    dx: float,
    dy: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process the finding of the best alignment angle.

    Args:
    ----
        acc_df (pd.DataFrame): The accelerometer data.
        angle_df (pd.DataFrame): The angle data.
        ground_truth (pd.DataFrame): The ground truth data.
        edit_map_dict (dict[str, np.ndarray]): The edit map dictionary.
        floor_name (str): The floor name.
        dx (float): The x-axis resolution.
        dy (float): The y-axis resolution.

    Returns:
    -------
        tuple[pd.DataFrame, pd.DataFrame]: The straight angle and straight angle displacement.

    """
    # 歩行タイミングの角度を求める
    angle_df_in_step_timing = utils.convert_to_peek_angle(
        acc_df,
        angle_df,
    )

    optimal_angle = _find_best_alignment_angle(
        angle_df_in_step_timing,
        ground_truth_df,
        edit_map_dict,
        floor_name,
        dx,
        dy,
    )

    straight_angle = pd.DataFrame(
        {
            "ts": angle_df["ts"],
            "x": angle_df["x"] + optimal_angle,
        },
    )

    straight_angle_displacement = (
        estimate.convert_to_peek_angle_and_compute_displacement_by_angle(
            straight_angle,
            acc_df,
            0.5,
            {"x": ground_truth_df.x[0], "y": ground_truth_df.y[0]},
            ground_truth_df["%time"][0],
        )
    )

    return straight_angle, straight_angle_displacement

from __future__ import annotations

from collections import deque
from typing import Literal

import estimate
import numpy as np
import pandas as pd
import utils


# その点が歩行可能かどうかを判断する関数
def is_passable(
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

    passable: bool = passable_dict[floor_name][row, col]

    return passable


def find_nearest_passable_point(
    passable_dict: dict[str, np.ndarray],
    floor_name: str,
    start_x: float,
    start_y: float,
    dx: float,
    dy: float,
) -> tuple[float, float] | None:
    start_row = int(start_x / dx)
    start_col = int(start_y / dy)

    queue = deque([(start_row, start_col)])
    visited: set[tuple[int, int]] = set()
    visited.add((start_row, start_col))

    if (
        start_row < 0
        or start_col < 0
        or start_row >= passable_dict[floor_name].shape[0]
        or start_col >= passable_dict[floor_name].shape[1]
    ):
        return None

    while queue:
        current_row, current_col = queue.popleft()

        if passable_dict[floor_name][current_row, current_col]:
            return current_row * dx, current_col * dy

        for neighbor_row, neighbor_col in [
            (current_row - 1, current_col),
            (current_row, current_col + 1),
            (current_row + 1, current_col),
            (current_row, current_col - 1),
        ]:
            if (
                0 <= neighbor_row < passable_dict[floor_name].shape[0]
                and 0 <= neighbor_col < passable_dict[floor_name].shape[1]
                and (neighbor_row, neighbor_col) not in visited
            ):
                queue.append((neighbor_row, neighbor_col))
                visited.add((neighbor_row, neighbor_col))

    return None


def correct_displacement(
    corrected_displacement_df: pd.DataFrame,
    map_dict: dict[str, np.ndarray],
    floor_name: str,
    index: int,
    nearest_row: pd.Series,
    dx: float,
    dy: float,
) -> pd.DataFrame:
    before_of_correction_point = {
        "x": nearest_row["x_displacement"],
        "y": nearest_row["y_displacement"],
    }

    corrected_point = find_nearest_passable_point(
        map_dict,
        floor_name,
        nearest_row["x_displacement"],
        nearest_row["y_displacement"],
        dx,
        dy,
    )
    if corrected_point is None:
        return corrected_displacement_df

    after_of_correction_point = {
        "x": corrected_point[0],
        "y": corrected_point[1],
    }

    delta_x = after_of_correction_point["x"] - before_of_correction_point["x"]
    delta_y = after_of_correction_point["y"] - before_of_correction_point["y"]

    corrected_displacement_df.loc[index:, ["x_displacement", "y_displacement"]] += [
        delta_x,
        delta_y,
    ]

    return corrected_displacement_df


Axis2D = Literal["x", "y"]


def move_unwalkable_points_to_walkable(
    acc_df: pd.DataFrame,
    angle_df: pd.DataFrame,
    map_dict: dict[str, np.ndarray],
    floor_name: str,
    dx: float,
    dy: float,
    *,
    ground_truth_first_point: dict[Axis2D, float] | None = None,
) -> pd.DataFrame:
    if ground_truth_first_point is None:
        ground_truth_first_point = {"x": 0.0, "y": 0.0}

    # 歩行タイミングの角度を求める
    angle_df_in_step_timing = utils.convert_to_peek_angle(
        acc_df,
        angle_df,
    )

    cumulative_displacement_df = (
        estimate.convert_to_peek_angle_and_compute_displacement_by_angle(
            angle_df_in_step_timing,
            acc_df,
            0.5,
            {
                "x": ground_truth_first_point["x"],
                "y": ground_truth_first_point["y"],
            },
        )
    )

    cumulative_displacement_df = cumulative_displacement_df.copy().reset_index(
        drop=True,
    )

    corrected_displacement_df = cumulative_displacement_df

    for index, _ in enumerate(cumulative_displacement_df.iterrows()):
        nearest_row = corrected_displacement_df.iloc[index]
        if not is_passable(
            map_dict,
            floor_name,
            nearest_row["x_displacement"],
            nearest_row["y_displacement"],
            dx,
            dy,
        ):
            corrected_displacement_df = correct_displacement(
                corrected_displacement_df,
                map_dict,
                floor_name,
                index,
                nearest_row,
                dx,
                dy,
            )

    return corrected_displacement_df

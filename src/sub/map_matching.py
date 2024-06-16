from __future__ import annotations

import heapq
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


# 幅優先探索による歩行可能座標の最短経路探索
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


# 斜め移動を考慮したダイクストラ法による最短経路探索
def find_nearest_passable_point_dijkstra(
    passable_dict: dict[str, np.ndarray],
    floor_name: str,
    start_x: float,
    start_y: float,
    dx: float,
    dy: float,
) -> tuple[float, float] | None:
    # グリッド座標に変換
    start_row = int(start_x / dx)
    start_col = int(start_y / dy)

    # 優先キューと訪問セットの初期化
    priority_queue = [(0, start_row, start_col)]  # (コスト, 行, 列)
    to_visit: set[tuple[int, int]] = set()
    to_visit.add((start_row, start_col))

    # 範囲チェック
    if (
        start_row < 0
        or start_col < 0
        or start_row >= passable_dict[floor_name].shape[0]
        or start_col >= passable_dict[floor_name].shape[1]
    ):
        return None

    # 方向と重みの設定
    directions = [
        ((-1, 0), 1),  # 上, 重み1
        ((1, 0), 1),  # 下, 重み1
        ((0, -1), 1),  # 左, 重み1
        ((0, 1), 1),  # 右, 重み1
        ((-1, -1), np.sqrt(2)),  # 左上, 重み√2
        ((-1, 1), np.sqrt(2)),  # 右上, 重み√2
        ((1, -1), np.sqrt(2)),  # 左下, 重み√2
        ((1, 1), np.sqrt(2)),  # 右下, 重み√2
    ]

    # 幅優先探索
    while priority_queue:
        current_cost, current_row, current_col = heapq.heappop(priority_queue)

        # 通行可能な点を見つけた場合
        if passable_dict[floor_name][current_row, current_col]:
            return current_row * dx, current_col * dy

        # 隣接する位置のチェック
        for direction, weight in directions:
            neighbor_row = current_row + direction[0]
            neighbor_col = current_col + direction[1]
            new_cost = current_cost + weight

            if (
                0 <= neighbor_row < passable_dict[floor_name].shape[0]
                and 0 <= neighbor_col < passable_dict[floor_name].shape[1]
                and (neighbor_row, neighbor_col) not in to_visit
            ):
                heapq.heappush(priority_queue, (new_cost, neighbor_row, neighbor_col))
                to_visit.add((neighbor_row, neighbor_col))

    # 通行可能な点が見つからない場合
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

    corrected_point = find_nearest_passable_point_dijkstra(
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
    ground_truth_first_point: dict[Axis2D, float],
) -> pd.DataFrame:
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


def calculate_centroid(
    coefficients: list[dict[Coefficient, float]],
) -> dict[str, float]:
    direction_error_sum = 0.0
    stride_length_error_sum = 0.0
    count = len(coefficients)

    for coefficient in coefficients:
        direction_error_sum += coefficient["direction_error_coefficient"]
        stride_length_error_sum += coefficient["stride_length_error_coefficient"]

    return {
        "direction_error_coefficient": direction_error_sum / count,
        "stride_length_error_coefficient": stride_length_error_sum / count,
    }


Coefficient = Literal["direction_error_coefficient", "stride_length_error_coefficient"]


def move_unwalkable_points_to_walkable2(
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

    # 歩幅の配列
    step_lengths = np.full(len(angle_df_in_step_timing), 0.5)
    befor_index = 0
    update_step_length = 0.5

    for index, _ in enumerate(cumulative_displacement_df.iterrows()):
        nearest_row = corrected_displacement_df.iloc[index]
        if index > 77:
            break
        if not is_passable(
            map_dict,
            floor_name,
            nearest_row["x_displacement"],
            nearest_row["y_displacement"],
            dx,
            dy,
        ):
            # 進行方向の誤差補正係数
            direction_error_coefficient = np.arange(0.9, 1.1, 0.01)
            # 歩幅の誤差補正係数
            stride_length_error_coefficient = np.arange(0.8, 1.2, 0.01)

            coefficient_combinations: list[dict[Coefficient, float]] = []

            for direction_error in direction_error_coefficient:
                for stride_length_error in stride_length_error_coefficient:
                    # 誤差補正後の累積変位を計算
                    calculate_cumulative_displacement_df = (
                        estimate.calculate_cumulative_displacement(
                            angle_df_in_step_timing["ts"],
                            angle_df_in_step_timing["x"] * direction_error,
                            update_step_length * stride_length_error,
                            {
                                "x": ground_truth_first_point["x"],
                                "y": ground_truth_first_point["y"],
                            },
                        )
                        .reset_index(drop=True)
                        .iloc[: index + 1]
                    )
                    # 誤差補正後の最後の点
                    specific_point = calculate_cumulative_displacement_df.iloc[index]

                    if is_passable(
                        map_dict,
                        floor_name,
                        specific_point["x_displacement"],
                        specific_point["y_displacement"],
                        dx,
                        dy,
                    ):
                        coefficient_combinations.append(
                            {
                                "direction_error_coefficient": direction_error,
                                "stride_length_error_coefficient": stride_length_error,
                            },
                        )

                        break

            centroid_coefficient = calculate_centroid(coefficient_combinations)

            # ここで軌跡を補正が完了したものに置き換える
            corrected_displacement_df = estimate.calculate_cumulative_displacement(
                angle_df_in_step_timing["ts"],
                angle_df_in_step_timing["x"]
                * centroid_coefficient["direction_error_coefficient"],
                update_step_length
                * centroid_coefficient["stride_length_error_coefficient"],
                {
                    "x": ground_truth_first_point["x"],
                    "y": ground_truth_first_point["y"],
                },
            ).reset_index(drop=True)

            # 軌跡全体の歩幅を更新する
            update_step_length = (
                update_step_length
                * centroid_coefficient["stride_length_error_coefficient"]
            )

            # 角度も更新する
            angle_df_in_step_timing["x"] = (
                angle_df_in_step_timing["x"]
                * centroid_coefficient["direction_error_coefficient"]
            )

    return corrected_displacement_df


def calculate_cumulative_displacement(
    ts: pd.Series,
    angle_data_x: pd.Series,
    step_lengths: np.ndarray,
    initial_point: dict[str, float],
    initial_timestamp: float = 0.0,
):
    x_displacement = step_lengths * np.cos(angle_data_x)
    y_displacement = step_lengths * np.sin(angle_data_x)

    init_data_frame = pd.DataFrame(
        {
            "ts": [initial_timestamp],
            "x_displacement": initial_point["x"],
            "y_displacement": initial_point["y"],
        },
    )

    return pd.concat(
        [
            init_data_frame,
            pd.DataFrame(
                {
                    "ts": ts,
                    "x_displacement": x_displacement.cumsum() + initial_point["x"],
                    "y_displacement": y_displacement.cumsum() + initial_point["y"],
                },
            ),
        ],
    )

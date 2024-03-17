

import pandas as pd
import numpy as np
import sys
import os
# fmt: off
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import estimate # noqa: E402
# fmt: on


# private functions
def _apply_remove_drift_to_angle(angle: pd.DataFrame, drift: float, base_time: pd.Timestamp) -> pd.DataFrame:
    """
    角度にドリフト除去を適用する
    """
    angle_with_drift = angle.copy()
    elapsed_time = angle_with_drift['ts'] - base_time
    angle_with_drift['x'] -= drift * elapsed_time
    return angle_with_drift

# private functions


def _compute_euclidean_distance(df: pd.DataFrame, gt: pd.Series) -> float:
    """
    軌跡の最後の座標と正解座標のユークリッド距離を計算する
    """
    last_row = df.iloc[-1]
    distance = np.sqrt((last_row['x_displacement'] - gt.x)**2 +
                       (last_row['y_displacement'] - gt.y)**2)
    return distance


# private functions
def _search_optimal_drift_from_angle(angle: pd.DataFrame, acc: pd.DataFrame, gt_ref: pd.DataFrame) -> dict:
    """
    角度のデータからドリフトを探索する
    """

    drift_range = np.arange(-0.05, 0.05, 0.001)
    base_time = angle['ts'].iloc[0]

    drift_and_euclidean_list = []
    for drift in drift_range:
        adjusted_angle = _apply_remove_drift_to_angle(angle, drift, base_time)
        displacement_df = estimate.convert_to_peek_angle_and_compute_displacement_by_angle(
            adjusted_angle, acc, 0.5, {"x": gt_ref.x[0], "y": gt_ref.y[0]}, gt_ref['%time'][0])
        displacement_df.reset_index(inplace=True, drop=True)

        euclidean_distance = _compute_euclidean_distance(
            displacement_df, gt_ref.iloc[1])
        drift_and_euclidean_list.append(
            {"drift": drift, "euclidean_distance": euclidean_distance})

    optimal_drift_and_euclidean = min(
        drift_and_euclidean_list, key=lambda x: x['euclidean_distance'])
    return optimal_drift_and_euclidean


# public functions
# 別のファイルからも呼び出される
def process_angle_data_with_drift_correction(angle_data: pd.DataFrame,
                                             acceleration_data: pd.DataFrame,
                                             ground_truth: pd.DataFrame) -> pd.DataFrame:
    """
    角度データを処理し、ドリフトを補正して変位を計算する。

    Args:
        angle_data: 角度データを含むDataFrame。
        acceleration_data: 加速度データを含むDataFrame。
        ground_truth: 基準となる位置データを含むDataFrame。

    Returns:
        ドリフト補正後の変位データを含むDataFrame。
    """
    # ドリフトの探索
    optimal_drift_and_euclidean = _search_optimal_drift_from_angle(
        angle_data, acceleration_data, ground_truth)

    # ドリフトの除去の適用
    corrected_angle_data = _apply_remove_drift_to_angle(angle_data,
                                                        optimal_drift_and_euclidean['drift'],
                                                        angle_data['ts'].iloc[0])

    # 変位の計算
    corrected_angle_displacement = estimate.convert_to_peek_angle_and_compute_displacement_by_angle(
        corrected_angle_data, acceleration_data,  0.5, {"x": ground_truth.x[0], "y": ground_truth.y[0]}, ground_truth['%time'][0])

    return corrected_angle_data, corrected_angle_displacement

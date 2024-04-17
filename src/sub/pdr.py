from __future__ import annotations

import os
import sys
from collections import defaultdict
from typing import Literal

import pandera as pa

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import estimate
import numpy as np
import pandas as pd
import utils
from PIL import Image
from scipy.signal import find_peaks

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

GIS_BASE_PATH = "../../gis/"
BEACON_LIST_PATH = GIS_BASE_PATH + "beacon_list.csv"
FLOOR_NAMES = ["FLU01", "FLU02", "FLD01"]
FOLDER_ID = "1qZBLQ66_pwRwLOy3Zj5q_qAwY_Z05HXb"


def _main() -> None:
    log_file_directory = "../../trials/"
    log_file_name = "4_1_51_pdr.txt"
    log_file_path = log_file_directory + log_file_name
    data = read_log_data(log_file_path)

    acc_df, gyro_df, _, ground_truth_df, _ = convert_to_dataframes(data)
    map_dict = load_floor_maps(FLOOR_NAMES, GIS_BASE_PATH)

    true_point: dict[Axis2D, float] = {
        "x": ground_truth_df["x"][0],
        "y": ground_truth_df["y"][0],
    }

    trajectory, _ = estimate_trajectory(
        acc_df,
        gyro_df,
        ground_truth_first_point=true_point,
    )
    utils.plot_displacement_map(map_dict, "FLU01", 0.01, 0.01, trajectory)


def read_log_data(log_file_path: str) -> dict:
    """Read log data from a file and return a dictionary.

    Args:
    ----
        log_file_path (str): The path to the log file.

    Returns:
    -------
        dict: A dictionary containing the parsed log data.

    """
    data = defaultdict(list)
    with open(log_file_path) as f:
        for line in f:
            line_contents = line.rstrip("\n").split(";")
            data_type = line_contents[0]
            if data_type == "BLUE":
                data["BLUE"].append(
                    {
                        "ts": float(line_contents[1]),
                        "bdaddress": line_contents[2],
                        "rssi": int(line_contents[4]),
                    },
                )
            elif data_type in ["ACCE", "GYRO", "MAGN"]:
                record = {
                    "ts": float(line_contents[1]),
                    "x": float(line_contents[3]),
                    "y": float(line_contents[4]),
                    "z": float(line_contents[5]),
                }
                data[data_type].append(record)
            elif data_type == "POS3":
                data["POS3"].append(
                    {
                        "ts": float(line_contents[1]),
                        "x": float(line_contents[3]),
                        "y": float(line_contents[4]),
                        "z": float(line_contents[5]),
                        "q0": float(line_contents[6]),
                        "q1": float(line_contents[7]),
                        "q2": float(line_contents[8]),
                        "q3": float(line_contents[9]),
                        "floor_name": line_contents[10],
                    },
                )

    return data


def convert_to_dataframes(
    data: dict,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Convert the given data dictionary into pandas DataFrames.

    Args:
    ----
        data (dict): The dictionary containing the data.

    Returns:
    -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: The converted DataFrames.

    """
    acc_df = pd.DataFrame(data["ACCE"])
    gyro = pd.DataFrame(data["GYRO"])
    mgf = pd.DataFrame(data["MAGN"])
    gt_ref = pd.DataFrame(data["POS3"])
    blescans = pd.DataFrame(data["BLUE"])

    acc_df = acc_df.reset_index(drop=True)
    gyro = gyro.reset_index(drop=True)
    mgf = mgf.reset_index(drop=True)
    gt_ref = gt_ref.reset_index(drop=True)

    # センサデータのスキーマを定義
    senser_schema = pa.DataFrameSchema(
        {
            "ts": pa.Column(pa.Float, nullable=False),
            "x": pa.Column(pa.Float, nullable=False),
            "y": pa.Column(pa.Float, nullable=False),
            "z": pa.Column(pa.Float, nullable=False),
        },
    )

    # ライダーデータのスキーマを定義
    lidar_schema = pa.DataFrameSchema(
        {
            "ts": pa.Column(pa.Float, nullable=False),
            "x": pa.Column(pa.Float, nullable=False),
            "y": pa.Column(pa.Float, nullable=False),
            "z": pa.Column(pa.Float, nullable=False),
            "q0": pa.Column(pa.Float, nullable=False),
            "q1": pa.Column(pa.Float, nullable=False),
            "q2": pa.Column(pa.Float, nullable=False),
            "q3": pa.Column(pa.Float, nullable=False),
            "floor_name": pa.Column(pa.String, nullable=False),
        },
    )

    # バリデーション
    senser_schema(acc_df)
    senser_schema(gyro)
    senser_schema(mgf)
    lidar_schema(gt_ref)

    return acc_df, gyro, mgf, gt_ref, blescans


def load_floor_maps(
    floor_names: list,
    base_path: str,
    optional_file_path: str = "",
) -> dict[str, np.ndarray]:
    """Load floor maps from the specified base path.

    Args:
    ----
        floor_names (list): List of floor names.
        base_path (str): Base path of the floor maps.
        optional_file_path (str): Optional file path to append to the base path.

    Returns:
    -------
        dict[str, np.ndarray]: Dictionary mapping floor names to floor maps.

    """
    map_dict: dict[str, np.ndarray] = {}
    for floor_name in floor_names:
        map_image_path = f"{base_path}{floor_name}_0.01_0.01{optional_file_path}.bmp"
        map_image = Image.open(map_image_path)
        map_dict[floor_name] = np.array(map_image, dtype=bool)
    return map_dict


def _process_sensor_data(
    acc_df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    # 加速度データのノルムを計算
    acc_df["norm"] = np.sqrt(acc_df["x"] ** 2 + acc_df["y"] ** 2 + acc_df["z"] ** 2)
    # ローリング平均によるノルムの平滑化
    acc_df["rolling_norm"] = acc_df["norm"].rolling(10).mean()
    # ステップ検出
    peaks, _ = find_peaks(acc_df["rolling_norm"], height=12, distance=10)
    return acc_df, peaks


Axis2D = Literal["x", "y"]


def estimate_trajectory(
    acc_df: pd.DataFrame,
    gyro_df: pd.DataFrame,
    *,
    ground_truth_first_point: dict[Axis2D, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate the trajectory using accelerometer and gyroscope data.

    Args:
    ----
        acc_df (pd.DataFrame): DataFrame containing accelerometer data.
        gyro_df (pd.DataFrame): DataFrame containing gyroscope data.
        ground_truth_first_point (dict[Axis2D, float] | None): Dictionary containing the first point of the ground truth trajectory.

    Returns:
    -------
        tuple[pd.DataFrame, pd.DataFrame]: Tuple containing the estimated trajectory and estimated angle.

    """
    if ground_truth_first_point is None:
        ground_truth_first_point = {"x": 0.0, "y": 0.0}

    acc_df, peaks = _process_sensor_data(acc_df)
    # ジャイロデータを用いてステップタイミングでの角度を推定
    peek_angle = estimate.convert_to_peek_angle(gyro_df, acc_df, peaks)
    # 累積変位の計算
    return estimate.calculate_cumulative_displacement(
        peek_angle.ts,
        peek_angle["x"],
        0.5,
        {
            "x": ground_truth_first_point["x"],
            "y": ground_truth_first_point["y"],
        },
        0.0,
    ), peek_angle


if __name__ == "__main__":
    _main()

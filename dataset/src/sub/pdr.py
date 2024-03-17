from __future__ import annotations

from collections import defaultdict

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
                    "accuracy": int(line_contents[6]),
                    "x": float(line_contents[3]),
                    "y": float(line_contents[4]),
                    "z": float(line_contents[5]),
                }
                data[data_type].append(record)
            elif data_type == "POS3":
                data["POS3"].append(
                    {
                        "%time": float(line_contents[1]),
                        "x": float(line_contents[3]),
                        "y": float(line_contents[4]),
                        "z": float(line_contents[5]),
                        "q0": float(line_contents[6]),
                        "q1": float(line_contents[7]),
                        "q2": float(line_contents[8]),
                        "q3": float(line_contents[9]),
                        "floor": line_contents[10],
                    },
                )

    return data


def _convert_to_dataframes(data: dict) -> tuple:
    acc = pd.DataFrame(data["ACCE"])
    gyro = pd.DataFrame(data["GYRO"])
    mgf = pd.DataFrame(data["MAGN"])
    gt_ref = pd.DataFrame(data["POS3"])

    acc = acc.reset_index(drop=True)
    gyro = gyro.reset_index(drop=True)
    mgf = mgf.reset_index(drop=True)
    gt_ref = gt_ref.reset_index(drop=True)

    return acc, gyro, mgf, gt_ref


def load_floor_maps(floor_names: list, base_path: str) -> dict[str, np.ndarray]:
    """Load floor maps from the specified base path.

    Args:
    ----
        floor_names (list): List of floor names.
        base_path (str): Base path of the floor maps.

    Returns:
    -------
        dict[str, np.ndarray]: Dictionary mapping floor names to floor maps.

    """
    map_dict: dict[str, np.ndarray] = {}
    for floor_name in floor_names:
        map_image_path = base_path + floor_name + "_0.01_0.01.bmp"
        map_image = Image.open(map_image_path)
        map_dict[floor_name] = np.array(map_image, dtype=bool)
    return map_dict


def _process_sensor_data(
    acc: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    # 加速度データのノルムを計算
    acc["norm"] = np.sqrt(acc["x"] ** 2 + acc["y"] ** 2 + acc["z"] ** 2)
    # ローリング平均によるノルムの平滑化
    acc["rolling_norm"] = acc["norm"].rolling(10).mean()
    # ステップ検出
    peaks, _ = find_peaks(acc["rolling_norm"], height=12, distance=10)
    return acc, peaks


def estimate_trajectory(
    acc: pd.DataFrame,
    gyro: pd.DataFrame,
    first_point: dict,
) -> pd.DataFrame:
    """Estimate the trajectory based on accelerometer and gyroscope data.

    Args:
    ----
        acc (pd.DataFrame): DataFrame containing accelerometer data.
        gyro (pd.DataFrame): DataFrame containing gyroscope data.
        first_point (dict): Dictionary containing the coordinates of the first point.

    Returns:
    -------
        pd.DataFrame: DataFrame containing the estimated trajectory.

    """
    acc, peaks = _process_sensor_data(acc)
    # ジャイロデータを用いてステップタイミングでの角度を推定
    peek_angle = estimate.convert_to_peek_angle(gyro, acc, peaks)
    # 累積変位の計算
    return estimate.calculate_cumulative_displacement(
        peek_angle.ts,
        peek_angle["x"],
        0.5,
        {"x": first_point["x"], "y": first_point["y"]},
        gt_ref["%time"][0],
    )


if __name__ == "__main__":
    log_file_directory = "../../trials/"
    log_file_name = "4_1_51_pdr.txt"
    log_file_path = log_file_directory + log_file_name
    data = read_log_data(log_file_path)
    acc, gyro, mgf, gt_ref = _convert_to_dataframes(data)
    map_dict = load_floor_maps(FLOOR_NAMES, "../../gis/")
    first_point = {"x": 0, "y": 0}
    trajectory = estimate_trajectory(acc, gyro, first_point)
    utils.plot_displacement_map_paper(map_dict, "FLU01", 0.01, 0.01, trajectory)

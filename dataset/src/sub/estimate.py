from collections import defaultdict, deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation as R

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

GIS_BASE_PATH = "./dataset/gis/"
BEACON_LIST_PATH = GIS_BASE_PATH + "beacon_list.csv"
FLOOR_NAMES = ["FLU01", "FLU02", "FLD01"]
FOLDER_ID = "1qZBLQ66_pwRwLOy3Zj5q_qAwY_Z05HXb"


def is_vertical_or_horizontal(angle, rotate_angle=0.0):
    rotated_angle = (angle + rotate_angle) % (2 * np.pi)

    is_vertical = (
        (rotated_angle >= np.pi / 2 - 0.1) & (rotated_angle <= np.pi / 2 + 0.1)
    ) | (
        (rotated_angle >= 3 * np.pi / 2 - 0.1) & (rotated_angle <= 3 * np.pi / 2 + 0.1)
    )

    is_horizontal = ((rotated_angle <= 0.1) | (rotated_angle >= 2 * np.pi - 0.1)) | (
        (rotated_angle >= np.pi - 0.1) & (rotated_angle <= np.pi + 0.1)
    )

    return is_vertical or is_horizontal


def convert_to_peek_angle_and_compute_displacement_by_angle(
    angle: pd.DataFrame,
    acc: pd.DataFrame,
    step_length: float,
    initial_point: dict[str, float],
    initial_timestamp: float = 0.0,
):
    peaks, _ = find_peaks(acc["rolling_norm"], height=12, distance=10)
    # 歩行タイミング時の角度をmatch_data関数を用いて取得
    angle_in_step_timing = match_data(angle, acc.ts[peaks])

    # 累積変位を計算
    cumulative_displacement_df = calculate_cumulative_displacement(
        angle_in_step_timing.ts,
        angle_in_step_timing["x"],
        step_length,
        initial_point,
        initial_timestamp,
    )

    cumulative_displacement_df["is_vertical_or_horizontal"] = angle_in_step_timing[
        "x"
    ].apply(
        is_vertical_or_horizontal,
    )

    return cumulative_displacement_df


def convert_to_peek_angle_and_compute_displacement_by_gyro(
    gyro: pd.DataFrame,
    acc: pd.DataFrame,
    peaks: np.ndarray,
    step_length: float,
    initial_point: dict[str, float],
    initial_timestamp: float = 0.0,
):
    # 角速度から歩行タイミングの角度を計算
    angle_in_step_timing = convert_to_peek_angle(gyro, acc, peaks)

    cumulative_displacement_df = calculate_cumulative_displacement(
        angle_in_step_timing.ts,
        angle_in_step_timing["x"],
        step_length,
        initial_point,
        initial_timestamp,
    )

    return cumulative_displacement_df


def match_data(something_df: pd.DataFrame, peek_t: pd.Series):
    matched_df = pd.DataFrame()
    for t in peek_t:
        matched_row = something_df[np.isclose(something_df["ts"], t, atol=0.005)]
        matched_df = pd.concat([matched_df, matched_row])
    return matched_df


def calculate_cumulative_displacement(
    ts: pd.Series,
    angle_data: pd.Series,
    step_length: float,
    initial_point: dict[str, float],
    initial_timestamp: float = 0.0,
):
    x_displacement = step_length * np.cos(angle_data)
    y_displacement = step_length * np.sin(angle_data)

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


def filter_strong_blescans(blescans: pd.DataFrame, ts: float, rssi: float):
    """Filter out blescans that have rssi value greater than -76 and ts value less than 30."""
    strong_blescans = blescans[(blescans["rssi"] > rssi) & (blescans["ts"] < ts)]
    strong_blescans.reset_index(inplace=True)
    return strong_blescans


def load_beacons_and_merge(BEACON_LIST_PATH, strong_blescans):
    df_beacons = pd.read_csv(BEACON_LIST_PATH)
    strong_blescans = pd.merge(strong_blescans, df_beacons, on="bdaddress", how="left")
    return strong_blescans


def compute_vector(point1: dict[str, np.float64], point2: dict[str, np.float64]):
    vector = {
        "x": point2["x"] - point1["x"],
        "y": point2["y"] - point1["y"],
    }
    return vector


def compute_angle_and_scale(vector1, vector2):
    dot = vector1["x"] * vector2["x"] + vector1["y"] * vector2["y"]
    norm_vector1 = np.sqrt(vector1["x"] ** 2 + vector1["y"] ** 2)
    norm_vector2 = np.sqrt(vector2["x"] ** 2 + vector2["y"] ** 2)

    scale = norm_vector2 / norm_vector1
    cos = dot / (norm_vector1 * norm_vector2)
    angle = np.arccos(cos)
    cross = vector1["x"] * vector2["y"] - vector1["y"] * vector2["x"]

    if cross < 0:
        angle = 2 * np.pi - angle

    return scale, angle


def affine_transform(
    displacement_df,
    center_point: dict[str, np.float64],
    scale,
    angle,
):
    displacement_df = displacement_df.copy()

    # 原点に平行移動
    displacement_df["x_displacement"] -= center_point["x"]
    displacement_df["y_displacement"] -= center_point["y"]

    # アフィン変換（拡大）
    displacement_df["x_displacement"] *= scale
    displacement_df["y_displacement"] *= scale

    # アフィン変換（半時計回転）
    new_x_displacement = displacement_df["x_displacement"] * np.cos(
        angle,
    ) - displacement_df["y_displacement"] * np.sin(angle)
    new_y_displacement = displacement_df["x_displacement"] * np.sin(
        angle,
    ) + displacement_df["y_displacement"] * np.cos(angle)

    displacement_df["x_displacement"] = new_x_displacement
    displacement_df["y_displacement"] = new_y_displacement

    #  平行移動（原点に移動させた分元に戻す）
    displacement_df["x_displacement"] += center_point["x"]
    displacement_df["y_displacement"] += center_point["y"]

    return displacement_df


def compute_affine_transformation(
    center_of_rotation_point: dict[str, np.float64],
    befor_of_correction_point: dict[str, np.float64],
    after_of_correction_point: dict[str, np.float64],
    cumulative_displacement_df: pd.DataFrame,
):
    cumulative_displacement_df = cumulative_displacement_df.copy()

    # centerからbeforeまでのベクトル
    vector_center_to_before = compute_vector(
        center_of_rotation_point,
        befor_of_correction_point,
    )

    # centerからafterまでのベクトル
    vector_center_to_after = compute_vector(
        center_of_rotation_point,
        after_of_correction_point,
    )

    scale, angle = compute_angle_and_scale(
        vector_center_to_before,
        vector_center_to_after,
    )

    cumulative_displacement_df = affine_transform(
        cumulative_displacement_df,
        center_of_rotation_point,
        scale,
        angle,
    )

    return cumulative_displacement_df


def unique_blescans_fn(strong_blescans):
    prev_item_control_number = strong_blescans["control_number"].values[0]
    unique_blescans = strong_blescans.iloc[[0]]
    # strong_blescansをfor文で回す
    for index, item in strong_blescans.iterrows():
        if item["control_number"] != prev_item_control_number:
            unique_blescans = pd.concat(
                [unique_blescans, strong_blescans.iloc[[index]]],
            )
            prev_item_control_number = item["control_number"]
    # unique_blescansの1行目を削除する
    unique_blescans = unique_blescans.iloc[1:]

    # control_numberが一意のものを抽出
    unique_blescans = unique_blescans[
        unique_blescans["bdaddress"].duplicated(keep=False) == False
    ]

    return unique_blescans


def filter_strongest_beacon_per_interval(df: pd.DataFrame, interval_seconds=10):
    """Filter beacon data by selecting the strongest signal in each interval.

    Args:
    ----
    df (pandas.DataFrame): DataFrame containing the beacon data.
    interval_seconds (int): Interval in seconds to group the data and select the strongest signal.

    Returns:
    -------
    filtered_df (pandas.DataFrame): DataFrame with filtered data.

    """
    # Create a new column for the interval index
    df["interval_index"] = (df["ts"] // interval_seconds).astype(int)

    # Group by the interval index and select the row with the maximum RSSI value in each group
    idx: pd.Series = df.groupby(["interval_index"])["rssi"].transform(max) == df["rssi"]
    filtered_df = df[idx]

    filtered_df = filtered_df.drop_duplicates(subset=["interval_index"])

    # Drop the interval index column
    filtered_df = filtered_df.drop(columns=["interval_index"])

    return filtered_df


def search_optimal_angle(
    displacement_df: pd.DataFrame,
    gt_ref: pd.DataFrame,
    strong_blescans: pd.DataFrame,
):
    # 探索する角度の範囲
    angle_range = np.arange(0, 2 * np.pi, 0.01)
    angle_and_euclidean_list: list[dict[str, float]] = []

    for angle in angle_range:
        new_df = rotate_cumulative_displacement(
            displacement_df,
            angle,
            {"x": gt_ref.x.iloc[0], "y": gt_ref.y.iloc[0]},
        )

        # Find nearest rows using merge_asof
        merged_df = pd.merge_asof(
            new_df.sort_values("ts"),
            strong_blescans.sort_values("ts"),
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


def search_optimal_drift_and_step_length(
    gyro: pd.DataFrame,
    acc: pd.DataFrame,
    gt_ref: pd.DataFrame,
    peaks: np.ndarray,
):
    gyro_copy = gyro.copy()

    # 探索するドリフトの範囲
    drift_range = np.arange(0, 0.3, 0.001)
    # 探索するステップ長の範囲
    step_length_range = np.arange(0.3, 0.6, 0.1)

    drift_and_euclidean_list: list[dict[str, float]] = []

    for step_length in step_length_range:
        # ドリフトを加えた角速度を計算
        for drift in drift_range:
            gyro_copy["new_x"] = gyro_copy["x"] - drift
            angle_copy = pd.DataFrame()
            # 角速度から角度に変換
            angle_copy["ts"] = gyro_copy["ts"]
            angle_copy["x"] = gyro_copy["new_x"].cumsum() * 0.01

            peek_angle_ = match_data(angle_copy, acc.ts[peaks])

            cumulative_displacement_df_copy = calculate_cumulative_displacement(
                peek_angle_.ts,
                peek_angle_["x"],
                step_length,
                {"x": gt_ref.x[0], "y": gt_ref.y[0]},
            )
            cumulative_displacement_df_copy.reset_index(inplace=True, drop=True)
            # cumulative_displacement_df_copyの最後の行の値を取得
            last_row = cumulative_displacement_df_copy.tail(1)
            # ユークリッド距離を計算
            euclidean_distance = np.sqrt(
                (last_row["x_displacement"].values[0] - 43) ** 2
                + (last_row["y_displacement"].values[0] - 12) ** 2,
            )

            drift_and_euclidean_list.append(
                {
                    "drift": drift,
                    "step_length": step_length,
                    "euclidean_distance": euclidean_distance,
                },
            )

    # 一番ユークリッド距離が小さいドリフトを取得
    optimal_drift_and_euclidean = min(
        drift_and_euclidean_list,
        key=lambda x: x["euclidean_distance"],
    )

    return optimal_drift_and_euclidean


def search_optimal_drift_from_gyro(
    gyro: pd.DataFrame,
    acc: pd.DataFrame,
    gt_ref,
    peaks: np.ndarray,
):
    gyro_copy = gyro.copy()

    # 探索するドリフトの範囲
    drift_range = np.arange(0, 0.3, 0.001)

    drift_and_euclidean_list: list[dict[str, float]] = []

    # ドリフトを加えた角速度を計算
    for drift in drift_range:
        gyro_copy["new_x"] = gyro_copy["x"] - drift
        angle_copy = pd.DataFrame()
        # 角速度から角度に変換
        angle_copy["ts"] = gyro_copy["ts"]
        angle_copy["x"] = gyro_copy["new_x"].cumsum() * 0.01

        peek_angle = match_data(angle_copy, acc.ts[peaks])

        cumulative_displacement_df_copy = calculate_cumulative_displacement(
            peek_angle.ts,
            peek_angle["x"],
            0.5,
            {"x": gt_ref.x[0], "y": gt_ref.y[0]},
        )
        cumulative_displacement_df_copy.reset_index(inplace=True, drop=True)
        # cumulative_displacement_df_copyの最後の行の値を取得
        last_row = cumulative_displacement_df_copy.tail(1)
        # ユークリッド距離を計算
        euclidean_distance = np.sqrt(
            (last_row["x_displacement"].values[0] - gt_ref.x[1]) ** 2
            + (last_row["y_displacement"].values[0] - gt_ref.y[1]) ** 2,
        )

        drift_and_euclidean_list.append(
            {"drift": drift, "euclidean_distance": euclidean_distance},
        )

    # 一番ユークリッド距離が小さいドリフトを取得
    optimal_drift_and_euclidean = min(
        drift_and_euclidean_list,
        key=lambda x: x["euclidean_distance"],
    )

    return optimal_drift_and_euclidean


def convert_to_peek_angle(gyro: pd.DataFrame, acc: pd.DataFrame, peaks: np.ndarray):
    gyro_copy = gyro.copy()
    angle_copy = pd.DataFrame()
    # 角速度から角度に変換
    angle_copy["ts"] = gyro_copy["ts"]
    angle_copy["x"] = gyro_copy["x"].cumsum() * 0.01

    peek_angle = match_data(angle_copy, acc.ts[peaks])

    # reset_index関数を使ってindexをリセット
    peek_angle = peek_angle.reset_index(drop=True)

    return peek_angle


def convert_to_angle_from_gyro(gyro: pd.DataFrame):
    gyro_copy = gyro.copy()
    angle_copy = pd.DataFrame()
    # 角速度から角度に変換
    angle_copy["ts"] = gyro_copy["ts"]
    angle_copy["x"] = gyro_copy["x"].cumsum() * 0.01
    angle_copy["y"] = gyro_copy["y"].cumsum() * 0.01
    angle_copy["z"] = gyro_copy["z"].cumsum() * 0.01

    return angle_copy


def convert_to_angle_from_cumulative_displacement(
    cumulative_displacement_df: pd.DataFrame,
):
    # コピーの作成
    cumulative_displacement_df_copy = cumulative_displacement_df.copy()
    # x方向とy方向の変位（累積変位の差分）を計算
    delta_x = cumulative_displacement_df_copy["x_displacement"].diff()
    delta_y = cumulative_displacement_df_copy["y_displacement"].diff()

    # 各ステップ間の相対角度を計算
    relative_angle = np.arctan2(delta_y, delta_x)

    # 1行目を削除
    relative_angle = relative_angle.drop(relative_angle.index[0])

    return pd.DataFrame(
        {
            # We drop the first timestamp due to diff
            "ts": cumulative_displacement_df_copy["ts"].iloc[1:],
            "x": relative_angle,
        },
    )


def convert_to_gyro(df: pd.DataFrame, sampling_frequency: float):
    # Calculate angle from displacement data
    angle_data = np.arctan2(df["y_displacement"], df["x_displacement"])
    # Calculate the difference between adjacent angle data
    angle_diff = np.diff(angle_data)

    # Convert the difference in angle to angular velocity
    gyro_data = angle_diff / sampling_frequency

    return pd.DataFrame(
        {
            "ts": df["ts"].iloc[1:],  # We drop the first timestamp due to diff
            "x": gyro_data,
        },
    )


def remove_drift_and_convert_to_angle(
    gyro: pd.DataFrame,
    acc: pd.DataFrame,
    drift: float,
    peaks: np.ndarray,
):
    gyro_copy = gyro.copy()
    # ドリフトが一定と仮定して除去
    gyro_copy["x"] = gyro_copy["x"] - drift
    angle_copy = pd.DataFrame()
    # 角速度から角度に変換
    angle_copy["ts"] = gyro_copy["ts"]
    angle_copy["x"] = gyro_copy["x"].cumsum() * 0.01
    peek_angle = match_data(angle_copy, acc.ts[peaks])

    return peek_angle


def search_optimal_drift_from_angle(angle: pd.DataFrame, gt_ref: pd.DataFrame):
    original_angle = angle.copy()
    # 探索するドリフトの範囲
    drift_range = np.arange(-0.01, 0.01, 0.001)
    drift_and_euclidean_list: list[dict[str, float]] = []

    for drift in drift_range:
        angle_copy = original_angle.copy()

        elapsed_time = angle_copy["ts"] - angle_copy["ts"].iloc[0]
        # ドリフトを経過時間に応じて増加させる
        angle_copy["x"] -= drift * elapsed_time

        cumulative_displacement_df_copy = calculate_cumulative_displacement(
            angle_copy.ts,
            angle_copy["x"],
            0.5,
            {"x": gt_ref.x[0], "y": gt_ref.y[0]},
        )
        cumulative_displacement_df_copy.reset_index(inplace=True, drop=True)
        # cumulative_displacement_df_copyの最後の行の値を取得
        last_row = cumulative_displacement_df_copy.tail(1)
        # ユークリッド距離を計算
        euclidean_distance = np.sqrt(
            (last_row["x_displacement"].values[0] - gt_ref.x[1]) ** 2
            + (last_row["y_displacement"].values[0] - gt_ref.y[1]) ** 2,
        )
        drift_and_euclidean_list.append(
            {"drift": drift, "euclidean_distance": euclidean_distance},
        )

    # 一番ユークリッド距離が小さいドリフトを取得
    optimal_drift_and_euclidean = min(
        drift_and_euclidean_list,
        key=lambda x: x["euclidean_distance"],
    )

    return optimal_drift_and_euclidean


# 画像の範囲外の場合はFalseを返すそれ以外の場合は座標のBooleanの値を返す


def is_passable(passable_dict, floor_name, x, y, dx, dy):
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

    passable = passable_dict[floor_name][row, col]
    return passable


def load_bitmap(filename):
    image = Image.open(filename)
    array = np.array(image, dtype=bool)
    return array


def plot_map(map_dict, floor_name, dx, dy):
    plt.figure(figsize=[10, 10])
    plt.axis("equal")

    # plot map
    xmax = map_dict[floor_name].shape[0] * dx  # length of map along x axis
    ymax = map_dict[floor_name].shape[1] * dy  # length of map along y axis

    plt.xlim(0, xmax)
    plt.ylim(0, ymax)

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(floor_name)
    plt.imshow(
        np.rot90(map_dict[floor_name]),
        extent=[0, xmax, 0, ymax],
        cmap="binary",
        alpha=0.5,
    )


def find_nearest_passable_point(passable_dict, floor_name, start_x, start_y, dx, dy):
    start_row = int(start_x / dx)
    start_col = int(start_y / dy)

    queue = deque([(start_row, start_col)])
    visited = set((start_row, start_col))

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


def correct_unpassable_points(
    cumulative_displacement_df: pd.DataFrame,
    map_dict,
    floor_name,
    dx,
    dy,
):
    cumulative_displacement_df = cumulative_displacement_df.copy().reset_index(
        drop=True,
    )
    corrected_displacement_df = cumulative_displacement_df

    for index, rows in cumulative_displacement_df.iterrows():
        nearest_row = corrected_displacement_df.iloc[index]

        if not is_passable(
            map_dict,
            floor_name,
            nearest_row["x_displacement"],
            nearest_row["y_displacement"],
            dx,
            dy,
        ):
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
                continue

            after_of_correction_point = {
                "x": corrected_point[0],
                "y": corrected_point[1],
            }

            delta_x = after_of_correction_point["x"] - before_of_correction_point["x"]
            delta_y = after_of_correction_point["y"] - before_of_correction_point["y"]

            # 平行移動
            corrected_displacement_df.loc[
                index:,
                ["x_displacement", "y_displacement"],
            ] += [delta_x, delta_y]

    return corrected_displacement_df


def find_best_alignment_angle(
    angle_df: pd.DataFrame,
    gt_ref: pd.DataFrame,
    edit_map_dict,
    floor_name,
    dx,
    dy,
):
    angle_range = np.arange(0, 2 * np.pi, 0.01)
    results = [
        calculate_horizontal_and_vertical_counts(angle_df, rotate_angle)
        for rotate_angle in angle_range
    ]
    df_results = pd.DataFrame(results).sort_values(
        by="horizontal_and_vertical_count",
        ascending=False,
    )
    df_results.reset_index(inplace=True, drop=True)

    df_results = calculate_exist_counts(
        angle_df,
        df_results,
        gt_ref,
        edit_map_dict,
        floor_name,
        dx,
        dy,
    )
    optimal_angle = get_optimal_angle(df_results)

    return optimal_angle


def calculate_horizontal_and_vertical_counts(
    angle_df: pd.DataFrame,
    rotate_angle: float,
) -> dict:
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


def calculate_exist_counts(
    angle_df: pd.DataFrame,
    results: pd.DataFrame,
    gt_ref: pd.DataFrame,
    edit_map_dict,
    floor_name,
    dx,
    dy,
):
    for i, row in results.head(20).iterrows():
        rotated_displacement = calculate_cumulative_displacement(
            angle_df.ts,
            (angle_df["x"] + row["angle"]),
            0.5,
            {"x": gt_ref.x[0], "y": gt_ref.y[0]},
            gt_ref["%time"][0],
        )

        exist_count = 0
        for _, displacement_row in rotated_displacement.iterrows():
            if is_passable(
                edit_map_dict,
                floor_name,
                displacement_row["x_displacement"],
                displacement_row["y_displacement"],
                dx,
                dy,
            ):
                exist_count += 1

        results.loc[i, "exist_count"] = exist_count

    return results


def get_optimal_angle(results: pd.DataFrame) -> float:
    max_exist_count = results["exist_count"].max()
    optimal_result = (
        results[results["exist_count"] == max_exist_count]
        .sort_values(by="horizontal_and_vertical_count", ascending=False)
        .iloc[0]
    )
    return optimal_result["angle"]


def extract_rotation(quaternions):
    res = R.from_quat(quaternions).apply([1, 0, 0])

    return np.arctan2(res[1], res[0])


def output_estimate_trajectory_include_ble(
    log_file_directory: str,
    log_file_name: str,
    output_directory: str,
    output_name: str,
):
    print(log_file_directory + log_file_name)
    # Prepare containers for the data
    data = defaultdict(list)
    with open(log_file_directory + log_file_name) as f:
        for line in f:
            line_contents = line.rstrip("\n").split(";")
            DATA_TYPE = line_contents[0]
            if DATA_TYPE == "BLUE":
                data["BLUE"].append(
                    {
                        "ts": float(line_contents[1]),
                        "bdaddress": line_contents[2],
                        "rssi": int(line_contents[4]),
                    },
                )

            elif DATA_TYPE in ["ACCE", "GYRO", "MAGN"]:
                record = {
                    "ts": float(line_contents[1]),
                    "accuracy": int(line_contents[6]),
                    "x": float(line_contents[3]),
                    "y": float(line_contents[4]),
                    "z": float(line_contents[5]),
                }
                data[DATA_TYPE].append(record)

            elif DATA_TYPE == "POS3":
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

    # Convert lists of dictionaries to dataframes
    blescans = pd.DataFrame(data["BLUE"])
    acc = pd.DataFrame(data["ACCE"])
    gyro = pd.DataFrame(data["GYRO"])
    mgf = pd.DataFrame(data["MAGN"])
    gt_ref = pd.DataFrame(data["POS3"])

    acc.reset_index(inplace=True, drop=True)
    gyro.reset_index(inplace=True, drop=True)
    mgf.reset_index(inplace=True, drop=True)
    gt_ref.reset_index(inplace=True, drop=True)

    acc["norm"] = np.sqrt(acc["x"] ** 2 + acc["y"] ** 2 + acc["z"] ** 2)

    # plt.plot(acc['ts'], acc['norm'])
    # plt.xlabel("time (s)")
    # plt.ylabel("acceleration norm (m/s^2)")
    # plt.title("Acceleration norm")
    # plt.savefig("./output/image/acc_norm/"+log_file_name[:-3]+"_acc_norm.png")
    # plt.figure(figsize=(15, 9))
    # plt.clf()  # 追加: 現在のプロットをクリア

    acc["rolling_norm"] = acc["norm"].rolling(10).mean()

    peaks, _ = find_peaks(acc["rolling_norm"], height=12, distance=10)

    # ジャイロを積分して角度に変換
    angle_in_step_timing = pd.DataFrame()
    angle_in_step_timing = convert_to_peek_angle(gyro, acc, peaks)

    # 1 pixel of bmp represents 0.01 m
    dx = 0.01
    dy = 0.01

    # read bitmap image of the floor movable areas
    map_dict = {}
    for floor_name in FLOOR_NAMES:
        map_dict[floor_name] = load_bitmap(
            GIS_BASE_PATH + floor_name + "_0.01_0.01.bmp",
        )

    floor_name = gt_ref["floor"].values[0]
    print("floor_name: ", floor_name)

    peek_angle = convert_to_peek_angle(gyro, acc, peaks)
    cumulative_displacement_df = calculate_cumulative_displacement(
        peek_angle.ts,
        peek_angle["x"],
        0.5,
        {"x": gt_ref.x[0], "y": gt_ref.y[0]},
        gt_ref["%time"][0],
    )

    # 軌跡前半の歩行軌跡の座標と強いBLEビーコンの位置座標の距離が最小になる角度を探索
    # これは軌跡前半はドリフトが乗りづらいため
    # 時間全体の中央を変数に入れる

    center_of_time = blescans.tail(1)["ts"].values[0] / 2
    first_half_strong_blescans = filter_strong_blescans(blescans, center_of_time, -76)
    merged_strong_blescans_first_half = load_beacons_and_merge(
        BEACON_LIST_PATH,
        first_half_strong_blescans,
    )

    angle = search_optimal_angle(
        cumulative_displacement_df,
        gt_ref,
        merged_strong_blescans_first_half,
    )

    rotate_by_first_half_angle = pd.DataFrame(
        {
            "ts": angle_in_step_timing.ts,
            "x": angle_in_step_timing.x + angle,
        },
    )

    rotate_by_first_half_angle_displacement = calculate_cumulative_displacement(
        rotate_by_first_half_angle.ts,
        rotate_by_first_half_angle["x"],
        0.5,
        {"x": gt_ref.x[0], "y": gt_ref.y[0]},
        gt_ref["%time"][0],
    )

    optimal_drift_and_euclidean = search_optimal_drift_from_angle(
        rotate_by_first_half_angle,
        gt_ref,
    )

    first_time_remove_drift_angle = pd.DataFrame(
        {
            "ts": rotate_by_first_half_angle.ts,
            "x": rotate_by_first_half_angle.x
            - optimal_drift_and_euclidean["drift"]
            * (rotate_by_first_half_angle.ts - rotate_by_first_half_angle.ts.iloc[0]),
        },
    )

    first_time_remove_drift_angle_displacement = calculate_cumulative_displacement(
        first_time_remove_drift_angle.ts,
        first_time_remove_drift_angle["x"],
        0.5,
        {"x": gt_ref.x[0], "y": gt_ref.y[0]},
        gt_ref["%time"][0],
    )

    # 軌跡全体の歩行軌跡の座標と強いBLEビーコンの位置座標の距離が最小になる角度を探索見つける
    overall_strong_blescans = filter_strong_blescans(
        blescans,
        blescans.tail(1)["ts"].values[0],
        -76,
    )
    merged_strong_blescans_overall = load_beacons_and_merge(
        BEACON_LIST_PATH,
        overall_strong_blescans,
    )

    overall_strong_ble_angle = search_optimal_angle(
        first_time_remove_drift_angle_displacement,
        gt_ref,
        merged_strong_blescans_overall,
    )

    rotate_by_overall_strong_ble_angle = pd.DataFrame(
        {
            "ts": first_time_remove_drift_angle.ts,
            "x": first_time_remove_drift_angle.x + overall_strong_ble_angle,
        },
    )

    second_optimal_drift_and_euclidean = search_optimal_drift_from_angle(
        rotate_by_overall_strong_ble_angle,
        gt_ref,
    )

    second_time_remove_drift_angle = pd.DataFrame(
        {
            "ts": rotate_by_overall_strong_ble_angle.ts,
            "x": rotate_by_overall_strong_ble_angle.x
            - second_optimal_drift_and_euclidean["drift"]
            * (
                rotate_by_overall_strong_ble_angle.ts
                - rotate_by_overall_strong_ble_angle.ts.iloc[0]
            ),
        },
    )

    second_time_remove_drift_angle_displacement = calculate_cumulative_displacement(
        second_time_remove_drift_angle.ts,
        second_time_remove_drift_angle["x"],
        0.5,
        {"x": gt_ref.x[0], "y": gt_ref.y[0]},
        gt_ref["%time"][0],
    )

    np.set_printoptions(threshold=np.inf)

    correct_unpassable_displacement = correct_unpassable_points(
        second_time_remove_drift_angle_displacement[
            second_time_remove_drift_angle_displacement["ts"] < 180
        ],
        map_dict,
        floor_name,
        dx,
        dy,
    )

    plot_map(
        map_dict,
        floor_name,
        dx,
        dy,
    )

    plt.colorbar(
        plt.scatter(
            correct_unpassable_displacement["x_displacement"],
            correct_unpassable_displacement["y_displacement"],
            c=correct_unpassable_displacement["ts"],
            cmap="rainbow",
            s=5,
        ),
    )

    plt.xlabel("x(m)")
    plt.ylabel("y(m)")
    plt.title("correct_unpassable_displacement")

    plt.savefig(f"{output_directory}image/{output_name}.png")

    # correct_unpassable_displacementにカラムを追加
    correct_unpassable_displacement["floor"] = floor_name

    correct_unpassable_displacement.to_csv(
        f"{output_directory}txt/{output_name}.csv",
        index=False,
        header=False,
    )

    print(f"{output_directory}txt/{output_name}.csv")


def output_estimate_trajectory_pdr(
    log_file_directory: str,
    log_file_name: str,
    output_directory: str,
    output_name: str,
):
    print(log_file_directory + log_file_name)
    # Prepare containers for the data
    data = defaultdict(list)
    with open(log_file_directory + log_file_name) as f:
        for line in f:
            line_contents = line.rstrip("\n").split(";")
            DATA_TYPE = line_contents[0]
            if DATA_TYPE == "BLUE":
                data["BLUE"].append(
                    {
                        "ts": float(line_contents[1]),
                        "bdaddress": line_contents[2],
                        "rssi": int(line_contents[4]),
                    },
                )

            elif DATA_TYPE in ["ACCE", "GYRO", "MAGN"]:
                record = {
                    "ts": float(line_contents[1]),
                    "accuracy": int(line_contents[6]),
                    "x": float(line_contents[3]),
                    "y": float(line_contents[4]),
                    "z": float(line_contents[5]),
                }
                data[DATA_TYPE].append(record)

            elif DATA_TYPE == "POS3":
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

    # Convert lists of dictionaries to dataframes
    acc = pd.DataFrame(data["ACCE"])
    gyro = pd.DataFrame(data["GYRO"])
    mgf = pd.DataFrame(data["MAGN"])
    gt_ref = pd.DataFrame(data["POS3"])

    acc.reset_index(inplace=True, drop=True)
    gyro.reset_index(inplace=True, drop=True)
    mgf.reset_index(inplace=True, drop=True)
    gt_ref.reset_index(inplace=True, drop=True)

    acc["norm"] = np.sqrt(acc["x"] ** 2 + acc["y"] ** 2 + acc["z"] ** 2)

    acc["rolling_norm"] = acc["norm"].rolling(10).mean()

    peaks, _ = find_peaks(acc["rolling_norm"], height=12, distance=10)

    # ジャイロを積分して角度に変換
    angle_in_step_timing = pd.DataFrame()
    angle_in_step_timing = convert_to_peek_angle(gyro, acc, peaks)

    # 1 pixel of bmp represents 0.01 m
    dx = 0.01
    dy = 0.01

    # read bitmap image of the floor movable areas
    map_dict = {}
    for floor_name in FLOOR_NAMES:
        map_dict[floor_name] = load_bitmap(
            GIS_BASE_PATH + floor_name + "_0.01_0.01.bmp",
        )

    edit_map_dict = {}
    for floor_name in FLOOR_NAMES:
        edit_map_dict[floor_name] = load_bitmap(
            GIS_BASE_PATH + floor_name + "_0.01_0.01_edited.bmp",
        )

    floor_name = gt_ref["floor"].values[0]
    print("floor_name: ", floor_name)

    peek_angle = convert_to_peek_angle(gyro, acc, peaks)
    cumulative_displacement_df = calculate_cumulative_displacement(
        peek_angle.ts,
        peek_angle["x"],
        0.5,
        {"x": gt_ref.x[0], "y": gt_ref.y[0]},
        gt_ref["%time"][0],
    )

    initial_direction = extract_rotation(
        [gt_ref.q0[0], gt_ref.q1[0], gt_ref.q2[0], gt_ref.q3[0]],
    )
    diff = angle_in_step_timing["x"].values[0] - initial_direction

    rotate_initial_direction_angle = pd.DataFrame(
        {
            "ts": angle_in_step_timing.ts,
            "x": angle_in_step_timing.x - diff,
        },
    )

    rotate_by_first_half_angle_displacement = calculate_cumulative_displacement(
        rotate_initial_direction_angle.ts,
        rotate_initial_direction_angle["x"],
        0.5,
        {"x": gt_ref.x[0], "y": gt_ref.y[0]},
        gt_ref["%time"][0],
    )

    optimal_drift_and_euclidean = search_optimal_drift_from_angle(
        rotate_initial_direction_angle,
        gt_ref,
    )

    first_time_remove_drift_angle = pd.DataFrame(
        {
            "ts": rotate_initial_direction_angle.ts,
            "x": rotate_initial_direction_angle.x
            - optimal_drift_and_euclidean["drift"]
            * (
                rotate_initial_direction_angle.ts
                - rotate_initial_direction_angle.ts.iloc[0]
            ),
        },
    )

    first_time_remove_drift_angle_displacement = calculate_cumulative_displacement(
        first_time_remove_drift_angle.ts,
        first_time_remove_drift_angle["x"],
        0.5,
        {"x": gt_ref.x[0], "y": gt_ref.y[0]},
        gt_ref["%time"][0],
    )

    angle_by_vertical_horizontal = find_best_alignment_angle(
        first_time_remove_drift_angle,
        gt_ref,
        edit_map_dict,
        floor_name,
        dx,
        dy,
    )

    rotate_veritcal_horizontal_angle = pd.DataFrame(
        {
            "ts": first_time_remove_drift_angle.ts,
            "x": first_time_remove_drift_angle.x + angle_by_vertical_horizontal,
        },
    )

    second_optimal_drift_and_euclidean = search_optimal_drift_from_angle(
        rotate_veritcal_horizontal_angle,
        gt_ref,
    )

    second_time_remove_drift_angle = pd.DataFrame(
        {
            "ts": rotate_veritcal_horizontal_angle.ts,
            "x": rotate_veritcal_horizontal_angle.x
            - second_optimal_drift_and_euclidean["drift"]
            * (
                rotate_veritcal_horizontal_angle.ts
                - rotate_veritcal_horizontal_angle.ts.iloc[0]
            ),
        },
    )

    second_time_remove_drift_angle_displacement = calculate_cumulative_displacement(
        second_time_remove_drift_angle.ts,
        second_time_remove_drift_angle["x"],
        0.5,
        {"x": gt_ref.x[0], "y": gt_ref.y[0]},
        gt_ref["%time"][0],
    )

    np.set_printoptions(threshold=np.inf)

    correct_unpassable_displacement = correct_unpassable_points(
        second_time_remove_drift_angle_displacement[
            second_time_remove_drift_angle_displacement["ts"] < 180
        ],
        map_dict,
        floor_name,
        dx,
        dy,
    )

    plot_map(
        map_dict,
        floor_name,
        dx,
        dy,
    )

    plt.colorbar(
        plt.scatter(
            correct_unpassable_displacement["x_displacement"],
            correct_unpassable_displacement["y_displacement"],
            c=correct_unpassable_displacement["ts"],
            cmap="rainbow",
            s=5,
        ),
    )

    plt.xlabel("x(m)")
    plt.ylabel("y(m)")
    plt.title("correct_unpassable_displacement")

    plt.savefig(f"{output_directory}image/{output_name}.png")

    # correct_unpassable_displacementにカラムを追加
    correct_unpassable_displacement["floor"] = floor_name

    # txtファイルに出力
    # headerはなし
    correct_unpassable_displacement.to_csv(
        f"{output_directory}txt/{output_name}.csv",
        index=False,
        header=False,
    )

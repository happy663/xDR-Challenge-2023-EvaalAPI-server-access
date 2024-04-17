import estimate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_stable_angle(
    angle_df: pd.DataFrame,
    stable_angle_range: float,
    stable_time: float,
) -> pd.DataFrame:
    stable_angle_df = pd.DataFrame()
    stable_angle_df["ts"] = angle_df["ts"]
    stable_angle_df["x"] = angle_df["x"]
    stable_angle_df["stable_flag"] = False

    start_row = stable_angle_df.iloc[0]
    elasted_time = 0
    stable_flag = False
    for index, row in stable_angle_df.iterrows():
        if (
            start_row["x"] - stable_angle_range
            <= row["x"]
            <= start_row["x"] + stable_angle_range
        ):
            elasted_time = row["ts"] - start_row["ts"]
            if elasted_time >= stable_time:
                stable_flag = True
        else:
            if stable_flag == True:
                stable_flag = False

                # stable_row.nameからrow.nameまでの平均を求める
                average = stable_angle_df.loc[start_row.name : row.name, "x"].mean()
                last_row_x = stable_angle_df.loc[row.name, "x"]

                # ステップ1: データの取得
                subset_data = stable_angle_df.loc[start_row.name : row.name, "x"]
                # ステップ2: 条件によるフィルタリング
                if last_row_x - average > 0:
                    filtered_values = subset_data[subset_data < average]
                else:
                    filtered_values = subset_data[subset_data > average]

                last_index = filtered_values.tail(1).index[0]

                if start_row.name - average > 0:
                    filtered_values = subset_data[subset_data < average]
                else:
                    filtered_values = subset_data[subset_data > average]

                first_index = filtered_values.head(1).index[0]

                # stable_angle_df.loc[start_row.name:row.name,'stable_flag']=True
                # stable_angle_df.loc[start_row.name:last_index,'stable_flag']=True
                stable_angle_df.loc[first_index:last_index, "stable_flag"] = True

            start_row = row

    return stable_angle_df


def correct_angle(stable_angle_df: pd.DataFrame, angle_column: str) -> pd.DataFrame:
    corrected_angle_df = stable_angle_df.copy()

    # 安定歩行区間での角度を修正する
    for index, row in stable_angle_df.iterrows():
        if row["stable_flag"]:
            angle = row[angle_column]
            # 最も近い安定歩行の角度を計算
            # 安定歩行の角度リスト（斜め方向も考慮）
            stable_angles = [
                0,
                1.5708,
                3.14159,
                4.71239,
                6.28319,
                7.85399,
                9.42478,
                10.9956,
                12.5664,
                14.1372,
                15.708,
                17.2788,
                18.8496,
                -1.5708,
                -3.14159,
                -4.71239,
                -6.28319,
                -7.85399,
                -9.42478,
                -10.9956,
                -12.5664,
                -14.1372,
                -15.708,
                -17.2788,
                -18.8496,
                0.7854,
                2.35619,
                -0.7854,
                -2.35619,
            ]
            closest_angle = min(stable_angles, key=lambda x: abs(x - angle))
            # 修正する
            corrected_angle_df.at[index, angle_column] = closest_angle

    return corrected_angle_df


def convert_to_peek_angle_and_compute_displacement_by_angle(
    angle: pd.DataFrame,
    acc: pd.DataFrame,
    step_length: float,
    initial_point: dict[str, float],
    initial_timestamp: float = 0.0,
):
    peaks, _ = estimate.find_peaks(acc["rolling_norm"], height=12, distance=10)
    # 歩行タイミング時の角度をmatch_data関数を用いて取得
    angle_in_step_timing = estimate.match_data(angle, acc.ts[peaks])

    # 累積変位を計算
    cumulative_displacement_df = calculate_cumulative_displacement(
        angle_in_step_timing["ts"],
        angle_in_step_timing,
        step_length,
        initial_point,
        initial_timestamp,
    )

    cumulative_displacement_df["is_vertical_or_horizontal"] = angle_in_step_timing[
        "x"
    ].apply(
        estimate.is_vertical_or_horizontal,
    )

    return cumulative_displacement_df


def calculate_cumulative_displacement(
    ts: pd.Series,
    angle_data: pd.DataFrame,
    step_length: float,
    initial_point: dict[str, float],
    initial_timestamp: float = 0.0,
):
    x_displacement = step_length * np.cos(angle_data["x"])
    y_displacement = step_length * np.sin(angle_data["x"])

    init_data_frame = pd.DataFrame(
        {
            "ts": [initial_timestamp],
            "x_displacement": initial_point["x"],
            "y_displacement": initial_point["y"],
            "stable_flag": angle_data["stable_flag"].values[0],
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
                    "stable_flag": angle_data["stable_flag"],
                },
            ),
        ],
    )


def plot_displacement_map(
    map_dict: dict[str, np.ndarray],
    floor_name: str,
    dx: float,
    dy: float,
    displacement_df: pd.DataFrame,
    *,
    fig_size: tuple[int, int] | None = None,
    display_map: bool = True,
    x_min: float = 0,
    y_min: float = 0,
    label_size: int = 10,
    font_size: int = 10,
    straight_angle_displacement: pd.DataFrame | None = None,  # 新しい引数
) -> None:
    """Plot a map with displacement data and optionally overlay additional scatter points."""
    if fig_size is not None:
        plt.figure(figsize=fig_size)
    else:
        plt.figure(figsize=(5, 5))

    xmax = map_dict[floor_name].shape[0] * dx
    ymax = map_dict[floor_name].shape[1] * dy

    plt.axis("equal")
    plt.xlim(x_min, xmax)
    plt.ylim(y_min, ymax)
    plt.xlabel("x (m)", fontsize=font_size)
    plt.ylabel("y (m)", fontsize=font_size)

    if display_map:
        plt.title(floor_name)
        plt.imshow(
            np.rot90(map_dict[floor_name]),
            extent=(0, xmax, 0, ymax),
            cmap="binary",
            alpha=0.5,
        )

    scatter = plt.scatter(
        displacement_df.x_displacement,
        displacement_df.y_displacement,
        c=displacement_df.ts,
        cmap="rainbow",
        s=5,
    )

    colorbar = plt.colorbar(scatter)
    colorbar.ax.tick_params(labelsize=label_size)
    colorbar.set_label("t (s)", fontsize=label_size)

    # 追加の散布図の描画（もしstraight_angle_displacementが提供された場合）
    if straight_angle_displacement is not None:
        plt.scatter(
            straight_angle_displacement.x_displacement[
                straight_angle_displacement["stable_flag"]
            ],
            straight_angle_displacement.y_displacement[
                straight_angle_displacement["stable_flag"]
            ],
            c="k",
            s=5,
        )

    plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt


# def refine_stable_regions_stricter(stable_angle_df, method='median', threshold_multiplier=0.5):
#     refined_df = stable_angle_df.copy()

#     # 安定歩行区間のインデックスを取得
#     stable_regions = []
#     start_idx = None
#     for idx, row in stable_angle_df.iterrows():
#         if row['stable_flag']:
#             if start_idx is None:
#                 start_idx = idx
#         else:
#             if start_idx is not None:
#                 stable_regions.append((start_idx, idx))
#                 start_idx = None

#     for start, end in stable_regions:
#         subset = stable_angle_df.loc[start:end, 'x']

#         # 指定された方法で中心値を計算
#         if method == 'mean':
#             center_value = subset.mean()
#         elif method == 'median':
#             center_value = subset.median()
#         elif method == 'mode':
#             center_value = subset.mode()[0]

#         # 中心値から大きく外れるデータを安定歩行区間から除外
#         threshold = threshold_multiplier * 0.35  # これは調整が必要かもしれません
#         outliers = (subset < center_value - threshold) | (subset > center_value + threshold)
#         refined_df.loc[outliers[outliers].index, 'stable_flag'] = False

#     return refined_df

# def is_stable_window(data, stable_angle_range):
#     # ウィンドウ内の最大値と最小値の差が安定範囲内に収まっているか判定
#     return (data.max() - data.min()) <= 2 * stable_angle_range

# def extract_stable_angle_with_sliding_window(angle_df: pd.DataFrame, stable_angle_range: float, stable_time: float) -> pd.DataFrame:
#     stable_angle_df = angle_df.copy()
#     stable_angle_df['stable_flag'] = False

#     window_size = int(stable_time / (angle_df['ts'].iloc[1] - angle_df['ts'].iloc[0]))  # ウィンドウサイズの計算（データポイント数）

#     for i in range(len(angle_df) - window_size + 1):
#         window_data = angle_df.iloc[i:i+window_size]['x']
#         if is_stable_window(window_data, stable_angle_range):
#             stable_angle_df.loc[angle_df.index[i:i+window_size], 'stable_flag'] = True

#     return stable_angle_df


# # スライドウィンドウを使用した関数を適用
# stable_angle_df = extract_stable_angle_with_sliding_window(straight_angle, 0.1, 3)
# # プロット
# plt.plot(stable_angle_df.ts, stable_angle_df['x'])
# plt.scatter(stable_angle_df['ts'][stable_angle_df['stable_flag']], stable_angle_df['x'][stable_angle_df['stable_flag']], c='r')
# plt.xlabel("timestamp (s)",fontsize=20)
# plt.ylabel("angle ($rad$)",fontsize=20)
# plt.show()

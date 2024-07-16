import numpy as np
import pandas as pd


# 安定区間を見つけるための関数
def find_stable_intervals(baro_df, pressure_col, time_col, threshold=0.03, duration=10):
    stable_intervals = [[]]
    start_index = 0
    window_size = int(duration)
    # duration秒に相当するデータポイント数
    stable_index = 0
    prev_is_stable = False

    while start_index + window_size < len(baro_df):
        window = baro_df.iloc[start_index : start_index + window_size]
        if (window[pressure_col].max() - window[pressure_col].min()) <= threshold:
            stable_intervals[stable_index].append(
                (window[time_col].iloc[0], window[time_col].iloc[-1]),
            )
            prev_is_stable = True
        else:
            if prev_is_stable:
                stable_intervals.append([])
                stable_index += 1
            prev_is_stable = False

        start_index += 1

    return stable_intervals


def process_baro_data(baro_df, stable_intervals):
    # Initialize list to hold dataframes for each interval group
    baro_df_list = []

    # Extract dataframes for each interval group
    for interval_group in stable_intervals:
        first_start, _ = interval_group[0]
        _, last_end = interval_group[-1]

        tmp_df = baro_df[
            (baro_df["ts"] >= first_start) & (baro_df["ts"] <= last_end)
        ].copy()
        tmp_df = tmp_df.reset_index(drop=True)
        baro_df_list.append(tmp_df)

    # Calculate mean for each dataframe
    baro_df_list_mean = [df["X (hPa)"].mean() for df in baro_df_list]

    # Sort the means
    sorted_means = sorted(baro_df_list_mean)

    # Group the means
    grouped_means = []
    current_group = [sorted_means[0]]
    for i in range(1, len(sorted_means)):
        if np.abs(sorted_means[i] - current_group[-1]) < 0.1:
            current_group.append(sorted_means[i])
        else:
            grouped_means.append(np.mean(current_group))
            current_group = [sorted_means[i]]
    if current_group:
        grouped_means.append(np.mean(current_group))

    # Create a dictionary of dataframes grouped by mean values
    grouped_lists = {
        idx: baro_df[abs(baro_df["X (hPa)"] - value) < 0.01].reset_index(drop=True)
        for idx, value in enumerate(grouped_means)
    }

    # Create a list of stable time intervals for each grouped list
    baro_df_stable_time_list = []
    for grouped_list in grouped_lists.values():
        stable_time_sublist = []
        first_index = 0
        for index, row in grouped_list.iterrows():
            if (
                index != 0
                and abs(row["ts"] - grouped_list.iloc[index - 1]["ts"]) > 10
                or index == len(grouped_list) - 1
            ):
                stable_time_sublist.append(
                    (
                        grouped_list.iloc[first_index]["ts"],
                        grouped_list.iloc[index - 1]["ts"],
                    ),
                )
                first_index = index
        baro_df_stable_time_list.append(stable_time_sublist)

    return (
        baro_df_list,
        baro_df_list_mean,
        grouped_means,
        grouped_lists,
        baro_df_stable_time_list,
    )


def highlight_stable_intervals(
    stable_interval: list,
    color: str,
    ax,
    trajectory_df: pd.DataFrame,
):
    for start, end in stable_interval:
        stable_points = trajectory_df[
            (trajectory_df["ts"] >= start) & (trajectory_df["ts"] <= end)
        ]
        ax.scatter(
            stable_points["x_displacement"],
            -stable_points["y_displacement"],
            color=color,
            s=3,
        )

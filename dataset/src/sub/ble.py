import numpy as np
import pandas as pd


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

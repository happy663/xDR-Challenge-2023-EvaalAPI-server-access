from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import numpy as np
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    import pandas as pd

from scipy.signal import find_peaks


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
) -> None:
    """Plot a map with displacement data.

    Parameters
    ----------
    - map_dict: Dictionary containing map data
    - floor_name: Name of the floor
    - dx: Width of each grid cell in meters
    - dy: Height of each grid cell in meters
    - displacement_df: DataFrame containing displacement data
    - fig_size: Size of the figure (optional)
    - display_map: Whether to display the map image (default: True)
    - x_min: Minimum x-coordinate value for the plot (default: 0)
    - y_min: Minimum y-coordinate value for the plot (default: 0)
    - label_size: Font size for labels (default: 10)
    - font_size: Font size for axis labels (default: 10)

    """
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
    plt.rcParams["font.size"] = font_size

    if display_map:
        plt.title(floor_name)
        plt.imshow(
            np.rot90(map_dict[floor_name]),
            extent=(0, xmax, 0, ymax),
            cmap="binary",
            alpha=0.5,
        )

    # 歩行軌跡の描画
    scatter = plt.scatter(
        displacement_df.x_displacement,
        displacement_df.y_displacement,
        c=displacement_df.ts,
        cmap="rainbow",
        s=1,
    )

    # カラーバーの追加
    colorbar = plt.colorbar(scatter)
    colorbar.ax.tick_params(labelsize=label_size)
    colorbar.set_label("t (s)", fontsize=label_size)

    # プロットの表示
    plt.show()


def plot_map(
    map_dict,
    floor_name,
    dx,
    dy,
    *,
    fig_size: tuple[int, int] | None = None,
    font_size: int = 10,
):
    if fig_size is not None:
        plt.figure(figsize=fig_size)
    else:
        plt.figure(figsize=(5, 5))
    plt.axis("equal")

    # plot map
    xmax = map_dict[floor_name].shape[0] * dx  # length of map along x axis
    ymax = map_dict[floor_name].shape[1] * dy  # length of map along y axis

    plt.xlim(0, xmax)
    plt.ylim(0, ymax)

    plt.xlabel("x (m)", fontsize=font_size)
    plt.ylabel("y (m)", fontsize=font_size)
    plt.title(floor_name, fontsize=font_size)

    plt.imshow(
        np.rot90(map_dict[floor_name]),
        extent=(0, xmax, 0, ymax),
        cmap="binary",
        alpha=0.5,
    )


def extract_rotation(quaternions: list[float]) -> float:
    """Extract the rotation angle from a quaternion.

    Parameters
    ----------
    quaternions : np.ndarray
        Array of quaternions representing rotations.

    Returns
    -------
    float
        The rotation angle in radians.

    """
    res = Rotation.from_quat(quaternions).apply([1, 0, 0])

    return np.arctan2(res[1], res[0])


def _match_data(something_df: pd.DataFrame, peek_t: pd.Series):
    matched_df = pd.DataFrame()
    for t in peek_t:
        matched_row = something_df[np.isclose(something_df["ts"], t, atol=0.005)]
        matched_df = pd.concat([matched_df, matched_row])
    return matched_df


def convert_to_peek_angle(acc_df: pd.DataFrame, angle_df: pd.DataFrame) -> pd.DataFrame:
    """Convert accelerometer data to peak angle data.

    Parameters
    ----------
    acc_df : pd.DataFrame
        DataFrame containing accelerometer data.
    angle_df : pd.DataFrame
        DataFrame containing angle data.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the matched data.

    """
    peaks, _ = find_peaks(acc_df["rolling_norm"], height=12, distance=10)
    # acc_dfのpeaksに対応するdfを表示
    peak_df = acc_df.iloc[peaks]

    return _match_data(angle_df, peak_df["ts"])

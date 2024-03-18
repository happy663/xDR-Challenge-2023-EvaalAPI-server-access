from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    import pandas as pd

# fmt: off
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# fmt: on


def plot_displacement_map(
    map_dict: dict[str, np.ndarray],
    floor_name: str,
    dx: float,
    dy: float,
    displacement_df: pd.DataFrame,
    x_min: float = 0,
    y_min: float = 0,
    label_size: int = 10,
    font_size: int = 10,
    *,
    display_map: bool = True,
) -> None:
    """Plot a map with displacement data.

    Parameters
    ----------
    - map_dict: Dictionary containing map data
    - floor_name: Name of the floor
    - dx: Width of each grid cell in meters
    - dy: Height of each grid cell in meters
    - displacement_df: DataFrame containing displacement data
    - x_min: Minimum x-coordinate value for the plot (default: 0)
    - y_min: Minimum y-coordinate value for the plot (default: 0)
    - display_map: Whether to display the map image (default: True)

    """
    xmax = map_dict[floor_name].shape[0] * dx
    ymax = map_dict[floor_name].shape[1] * dy

    plt.figure(figsize=[10, 10])
    plt.axis("equal")
    plt.xlim(x_min, xmax)
    plt.ylim(y_min, ymax)
    plt.xlabel("x (m)", fontsize=font_size)
    plt.ylabel("y (m)", fontsize=font_size)

    if display_map:
        plt.title(floor_name)
        plt.imshow(
            np.rot90(map_dict[floor_name]),
            extent=[0, xmax, 0, ymax],
            cmap="binary",
            alpha=0.5,
        )

    # 歩行軌跡の描画
    scatter = plt.scatter(
        displacement_df.x_displacement,
        displacement_df.y_displacement,
        c=displacement_df.ts,
        cmap="rainbow",
        s=5,
    )

    # カラーバーの追加
    colorbar = plt.colorbar(scatter)
    colorbar.ax.tick_params(labelsize=label_size)
    colorbar.set_label("t (s)", fontsize=label_size)

    # プロットの表示
    plt.show()


# これは軌跡前半はドリフトが乗りづらいため
# 時間全体の中央を変数に入れる
def extract_rotation(quaternions: np.ndarray) -> float:
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
    res = R.from_quat(quaternions).apply([1, 0, 0])

    return np.arctan2(res[1], res[0])

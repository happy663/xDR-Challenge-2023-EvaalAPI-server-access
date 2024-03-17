import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

# fmt: off
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import estimate  # noqa: E402

# fmt: on


def _plot_map(map_dict, floor_name, dx, dy):
    plt.figure(figsize=[10, 10])
    plt.axis("equal")

    # plot map
    xmax = map_dict[floor_name].shape[0] * dx  # length of map along x axis
    ymax = (
        map_dict[floor_name].shape[1] * dy
    )  # length of map aloneq       qryuiog y axis

    plt.xlim(-20, xmax)
    plt.ylim(0, ymax)

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    # plt.title(floor_name)
    # plt.imshow(np.rot90(map_dict[floor_name]),
    # extent=[0, xmax, 0, ymax], cmap="binary", alpha=0.5)


def plot_displacement_map(map_dict, floor_name, dx, dy, displacement_df):
    """
    Plot a map with displacement data.

    Parameters:
    - map_dict: Dictionary containing map data
    - floor_name: Name of the floor
    - dx, dy: Dimensions for the plot
    - displacement_df: DataFrame containing displacement data
    """
    # マップの描画
    _plot_map(map_dict, floor_name, dx, dy)
    # 散布図の描画
    plt.scatter(
        displacement_df.x_displacement,
        displacement_df.y_displacement,
        c=displacement_df.ts,
        cmap="rainbow",
        s=5,
    )
    # カラーバーの追加
    # plt.colorbar()

    # プロットの表示
    plt.show()


# 論文の図で表示する用
def plot_displacement_map_paper(map_dict, floor_name, dx, dy, displacement_df):
    """
    Plot a map with displacement data.

    Parameters:
    - map_dict: Dictionary containing map data
    - floor_name: Name of the floor
    - dx, dy: Dimensions for the plot
    - displacement_df: DataFrame containing displacement data
    """
    print(map_dict[floor_name].shape[0])
    # マップの描画
    _plot_map(map_dict, floor_name, dx, dy)

    # 散布図の描画
    scatter = plt.scatter(
        displacement_df.x_displacement,
        displacement_df.y_displacement,
        c=displacement_df.ts,
        cmap="rainbow",
        s=5,
    )
    # カラーバーの追加
    # plt.colorbar()

    cbar = plt.colorbar(scatter)

    # colorbarの文字サイズを大きく
    cbar.ax.tick_params(labelsize=20)
    # 単位を表示
    cbar.set_label("t (s)", fontsize=20)

    # xlabelの文字サイズを大きく
    plt.tick_params(labelsize=20)
    plt.xlabel("x (m)", fontsize=20)
    plt.ylabel("y (m)", fontsize=20)

    # プロットの表示
    plt.show()


# 軌跡前半の歩行軌跡の座標と強いBLEビーコンの位置座標の距離が最小になる角度を探索
# これは軌跡前半はドリフトが乗りづらいため
# 時間全体の中央を変数に入れる
def extract_rotation(quaternions):
    res = R.from_quat(quaternions).apply([1, 0, 0])

    return np.arctan2(res[1], res[0])

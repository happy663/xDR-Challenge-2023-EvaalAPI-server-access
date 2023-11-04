import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


LOG_FILE_PATH = "../trials/4_1_51.txt"
GT_CSV_PATH = "../gt/4_1_gt.csv"

GIS_BASE_PATH = "../gis/"
BEACON_LIST_PATH = GIS_BASE_PATH + "beacon_list.csv"
FLOOR_NAMES = ["FLU01", "FLU02", "FLD01"]
FOLDER_ID = "1qZBLQ66_pwRwLOy3Zj5q_qAwY_Z05HXb"


# 正解軌跡を出力する関数
def output_gt(gt_filepath: str):

    # This cell shows BLE beacons and map information
    def load_bitmap(filename):
        image = Image.open(filename)
        array = np.array(image, dtype=bool)
        return array

    # 1 pixel of bmp represents 0.01 m
    dx = 0.01
    dy = 0.01

    # read bitmap image of the floor movable areas
    map_dict = {}
    for floor_name in FLOOR_NAMES:
        map_dict[floor_name] = load_bitmap(
            GIS_BASE_PATH + floor_name + "_0.01_0.01.bmp")

    # read the beacon list
    df_beacons = pd.read_csv(BEACON_LIST_PATH)

    df_gt = pd.read_csv(gt_filepath)

    # get the floor name in the ground truth
    floor_name = df_gt.floor.mode()[0]

    # visualize the gis data
    plt.figure(figsize=[10, 10])
    plt.axis("equal")

    # plot map
    xmax = map_dict[floor_name].shape[0] * dx  # length of map along x axis
    ymax = map_dict[floor_name].shape[1] * dy  # length of map along y axis
    plt.imshow(np.rot90(map_dict[floor_name]),
               extent=[0, xmax, 0, ymax], cmap="binary", alpha=0.5)

    # plot beacons
    # extract beacons in the designated floor
    beacons_in_floor = df_beacons[df_beacons["floorname"] == floor_name]
    plt.scatter(beacons_in_floor.x, beacons_in_floor.y, c="b", label="beacon")
    plt.legend()

    # plot ground truth path
    cm = plt.scatter(df_gt.x, df_gt.y, c=df_gt["%time"], s=1, cmap="rainbow")
    plt.colorbar(cm, label="timestamps (s)")

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(floor_name)

    plt.savefig("./output/image/"+gt_filepath[5:-4]+".png")

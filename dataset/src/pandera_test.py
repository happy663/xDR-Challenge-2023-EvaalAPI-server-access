import importlib

from sub import pdr

LOG_FILE_PATH = "../trials/4_1_51_pdr.txt"
GIS_BASE_PATH = "../gis/"
BEACON_LIST_PATH = GIS_BASE_PATH + "beacon_list.csv"
FLOOR_NAMES = ["FLU01", "FLU02", "FLD01"]
importlib.reload(pdr)

data = pdr.read_log_data(LOG_FILE_PATH)
acc_df, gyro_df, mgf_df, ground_truth_df, blescans_df = pdr.convert_to_dataframes(data)
print(acc_df.head())
print(gyro_df.head())
print(mgf_df.head())
print(ground_truth_df.head())
print(blescans_df.head())

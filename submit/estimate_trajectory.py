import sys

from src.sub import estimate

if __name__ == "__main__":
    REQUIRED_ARGUMENTS = 3
    if len(sys.argv) != REQUIRED_ARGUMENTS:
        print(
            "使用方法: python3 estimate_trajectory_include_ble.py <trial_file_without_extension>"
            " <mode: pdr or ble>",
        )
        sys.exit(1)

    trial_filename_base = sys.argv[1]
    mode = sys.argv[2]

    if mode == "pdr":
        estimate.output_estimate_trajectory_pdr(
            "./dataset/trials/",
            trial_filename_base + "_pdr.txt",
            "./dataset/traj/",
            trial_filename_base + "_pdr_est",
        )
    elif mode == "ble":
        estimate.output_estimate_trajectory_include_ble(
            "./dataset/trials/",
            trial_filename_base + ".txt",
            "./dataset/traj/",
            trial_filename_base + "_est",
        )

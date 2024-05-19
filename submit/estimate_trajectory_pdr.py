import sys
from dataset.src.sub import estimate

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("使用方法: python3 estimate_trajectory_include_ble.py <trial_file_without_extension>")
        sys.exit(1)

    trial_filename_base = sys.argv[1]

    estimate.output_estimate_trajectory_pdr(
        './dataset/trials/', trial_filename_base +
        '_pdr.txt', './dataset/traj/', trial_filename_base + '_pdr_est'
    )

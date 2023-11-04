

import sys
from dataset.src.sub import estimate

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("使用方法: python3 estimate_trajectory_include_ble.py <trial_file_without_extension>")
        sys.exit(1)

    trial_filename_base = sys.argv[1]

    estimate.output_estimate_trajectory_include_ble(
        './dataset/trials/', trial_filename_base +
        '.txt', './dataset/traj/', trial_filename_base + '_est'
    )

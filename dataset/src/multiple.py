import sub.estimate as estimate
import datetime
from sub import google_drive
import os
import re
import single


def get_filenames_in_directory(directory_path):
    filenames = os.listdir(directory_path)
    return filenames


if __name__ == '__main__':

    now = datetime.datetime.now()
    # yy/MM/dd H:mm
    formatted_now = now.strftime('%m月%d日%H時%M分')
    created_folder_id = google_drive.create_folder(
        formatted_now)

    filenames = get_filenames_in_directory("../trials")
    # 結果を表示
    fileter_filenames = [
        filename for filename in filenames if "pdr" not in filename]
    # fileter_filenamesの数だけfor文を回す
    for fileter_filename in fileter_filenames:

        print(fileter_filename)

        match = re.match(r'(\d+_\d+)_\d+.txt', fileter_filename)

        if match is not None:
            extracted_part = re.match(r'(\d+_\d+)_\d+.txt',
                                      fileter_filename).groups()[0]
            print(extracted_part)

            gt_file_path = "../gt/"+extracted_part+"_gt.csv"

            # estimate.output_estimate_trajectory_include_ble(
            #     "../trials/", fileter_filename, gt_file_path
            # )

            single.single_run("../trials/", fileter_filename,
                              gt_file_path, created_folder_id)
        else:
            print("not match")

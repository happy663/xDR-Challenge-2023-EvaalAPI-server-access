import sub.gt as gt
import sub.estimate as estimate
import datetime
from sub import google_drive
import os


def single_run(log_file_directory: str, log_file_name: str, gt_filepath: str, created_folder_id: str):

    # 推定軌跡の出力とアップロード
    estimate.output_estimate_trajectory_include_ble(
        log_file_directory, log_file_name
    )

    google_drive.upload_to_drive(
        log_file_name[:-3]+'.png', './output/image/' +
        log_file_name[:-3]+'.png', 'image/jpeg', created_folder_id
    )

    # # 正解軌跡の画像が存在しない場合は作成
    if not os.path.exists('./output/image/'+gt_filepath[6:-4]+'.png'):
        gt.output_gt(gt_filepath)
        google_drive.upload_to_drive(
            gt_filepath[6:-4]+'.png', './output/image/' +
            gt_filepath[6:-4]+'.png', 'image/jpeg', created_folder_id
        )
    else:
        print("already exists")


if __name__ == "__main__":
    # # 現在の日時を取得
    now = datetime.datetime.now()
    # yy/MM/dd H:mm
    formatted_now = now.strftime('%m月%d日%H時%M分')
    created_folder_id = google_drive.create_folder(
        formatted_now
    )
    single_run('../trials/', '8_1_51.txt',
               '../gt/8_1_gt.csv', created_folder_id)

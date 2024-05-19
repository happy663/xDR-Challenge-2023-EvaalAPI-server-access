
#!/bin/bash

# 引数を変数にセット
TRIAL_ID=$1

if [ -z "$TRIAL_ID" ]; then
    echo "エラー: TRIAL_IDが必要です。"
    exit 1
fi


python3 do_downloading_trials.py $TRIAL_ID http://121.196.218.26/evaalapi/ ./dataset/trials/${TRIAL_ID}_pdr.txt
python3 estimate_trajectory_pdr.py $TRIAL_ID
python3 do_submitting_estimates.py $TRIAL_ID http://121.196.218.26/evaalapi/ ./dataset/traj/txt/${TRIAL_ID}_pdr_est.txt


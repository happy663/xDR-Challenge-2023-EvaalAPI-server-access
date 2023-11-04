#!/bin/bash

# 引数を変数にセット
TRIAL_ID=$1
MODE=$2

if [ -z "$TRIAL_ID" ]; then
    echo "エラー: TRIAL_IDが必要です。"
    exit 1
fi

if [ -z "$MODE" ]; then
    echo "エラー: MODEが必要です。(ble または pdr)"
    exit 1
fi


SERVER_URL=http://121.196.218.26/evaalapi/

if [ "$MODE" == "pdr" ]; then
    # python3 do_downloading_trials.py $TRIAL_ID $SERVER_URL ./dataset/trials/${TRIAL_ID}_pdr.txt
    python3 estimate_trajectory.py $TRIAL_ID $MODE
    # python3 do_submitting_estimates.py $TRIAL_ID $SERVER_URL ./dataset/traj/txt/${TRIAL_ID}_pdr_est.txt
elif [ "$MODE" == "ble" ]; then
    # python3 do_downloading_trials.py $TRIAL_ID $SERVER_URL ./dataset/trials/${TRIAL_ID}.txt
    python3 estimate_trajectory.py $TRIAL_ID $MODE
    # python3 do_submitting_estimates.py $TRIAL_ID $SERVER_URL ./dataset/traj/txt/${TRIAL_ID}_est.txt
fi


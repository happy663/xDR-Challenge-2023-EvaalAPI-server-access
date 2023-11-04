# xDR-Challenge-2023-EvaalAPI-server-access
The details how to download the testing or scoring trials of xDR Challenge 2023 (IPIN 2023 competition track5) and submit the estimates generated at your side using Evaal API server are introduced in https://evaal.aaloa.org/2023/software-for-offsite-tracks. 
This README explains the distributed sample scripts. These scripts are successful examples of download / submittion, therefore you can check and demonstrate the operation of  download / submittion as well as the associating EvAAL API server's behavior. 

Note that testing or scoring "trial names" are necessary for the download and submission steps. Please ask for trial names to info@evaal.aaloa.org in advance.


## Requirements (Required python version and packages for running the scripts)
```
requests==2.29.0
```

## Description of Files

| **Filename** | **Description** |
 ---            |---
| do_downloading_trials.py | Execute to dowload a trial |
| do_submitting_estimates.py | Execute to submit your estimates |
| requirements.txt        | File for summarizing the requirements|

## Usage

Please proceed following steps after having trial names.

### Step.1  Install
```
git clone --recursive https://github.com/PDR-benchmark-standardization-committee/xDR-Challenge-2023-download-submission
cd xDR-Challenge-2023-download-submission
pip install -r requirements.txt
```

### Step.2 Placing scripts and folders
Please place folders as the example below.
The folder path is necessary to be written correctly in commands for download / submission instructed later.
```
xDR-Challenge-2023-evaluation/
├ dataset/
|   ├ traj/     (to be used for submission)
|   └ trials/   (to be used for download)
|
├ do_downloading_trials.py
├ do_submitting_estimates.py
├ requirements.txt
└ README.md
```

### Step.3 Downloading a trial
One trial name can be used for downloading one corresponding trial data and submitting one corresponding estimates.
Execute following script to download a trial data.
```
python do_downloading_trials.py [trial_name] [server_url] ./dataset/trials/[give_the_name_as_you_like].txt
```
The trial data will be saved in [output] folder with receiving respone code 200.
See Data Format section in the readme on the official web site (https://unit.aist.go.jp/harc/xDR-Challenge-2023/data/README.md).

### Step.4 Trajectory estimation
Run your own script and generate your estimated trajectory file.
The contents of the estimated trajectory file should be separated by commas as follows.
```
Timestamp(s),x(m),y(m),floor(FLU01/FLU02/FLD01)
```
The latest dataset provided via official web site (https://unit.aist.go.jp/harc/xDR-Challenge-2023/data/xdrchallenge2023_dev_0712.zip) includes a demo script (02_output_example.ipynb) supposing to output estimated trajectory files that follow this file format. But make sure to satisfy following notes at your side;
- Headers should not be included in the trajectory file.
- All lines should be sorted in ascending order by timestamp. Timestamps with reverse order causes the submission to be rejected.
- Each timestamp must be wrirtten in fixed-point notation and not be a negative value. Using exponential notation for timestamps or being a negative value causes the submission to be rejected.

### Step.5 Submitting the estimated trajectories
Please place (copy) the file of estimated trajectory at [dataset]/[traj]/.
Then, execute following script to submit the estimated trajectory.
```
python do_submitting_estimates.py [trial_name] [server_url] ./dataset/traj/[your_file_name_of_estimated trajectory].txt
```
The submitted data will be accepted by the server with receiving respone code 201.

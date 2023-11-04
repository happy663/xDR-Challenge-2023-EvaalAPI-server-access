# -*- coding: utf-8 -*-
import requests
import time
import sys

# HTTP GET request


def do_get(req):
    r = requests.get(req)
    print('==>  GET ' + req + ' --> ' + str(r.status_code))
    return (r)

# HTTP POST request


def do_post(req, data, headers):
    r = requests.post(req, data=data, headers=headers)
    print(r.text)
    print('==>  POST ' + req + ' --> ' + str(r.status_code))
    return (r)

# Read file data


def read_file(file_path):
    with open(file_path) as f:
        text_data = f.read()
    return text_data


def main(trialname, server, file_path, w):
    url = server + trialname
    estimates = read_file(file_path)        # Load the estimates
    time.sleep(w)
    r = do_get(url + '/state')              # Trial state confirmation
    time.sleep(w)
    r = do_post(url + '/estimates', data=estimates,
                headers={'content-type': 'text/csv; charset=us-ascii'})     # Submit the estimates


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage is [trial name] [server URL] [path_to_estimate]')
        exit(1)

    w = 3                                   # Wait for completion of previous process

    TRIAL_NAME = sys.argv[1]
    SERVER_URL = sys.argv[2]
    SUBMIT_FILE_PATH = sys.argv[3]
    main(TRIAL_NAME, SERVER_URL, SUBMIT_FILE_PATH, w)

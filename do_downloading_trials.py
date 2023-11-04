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

# Save data to file


def write_file(text_data, file_path):
    with open(file_path, mode='w') as f:
        f.write(text_data)
    f.close()


def main(trialname, server, file_path, w):
    url = server + trialname
    # Set the trial state to 'nonstarted'.
    r = do_get(url + '/reload')
    time.sleep(w)
    r = do_get(url + '/state')              # Trial state confirmation
    time.sleep(w)
    # Get data from the EvAAL API server
    r = do_get(url + '/nextdata?offline')
    time.sleep(w)
    write_file(r.text, file_path)           # Save the received data


if __name__ == '__main__':

    if len(sys.argv) != 4:
        print('Usage is [trial name] [server URL] [path_to_save_trial]')
        exit(1)

    w = 3                                   # Waiting for completion of previous process

    TRIAL_NAME = sys.argv[1]
    SERVER_URL = sys.argv[2]
    TRIAL_FILE_PATH = sys.argv[3]
    main(TRIAL_NAME, SERVER_URL, TRIAL_FILE_PATH, w)

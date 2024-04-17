import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

# OAuth2.0の認証情報のJSONファイルのパス
CREDENTIALS_FILE = "../../secret/credentials.json"
# トークンの保存場所
TOKEN_FILE = "token.json"

# 親フォルダのID
PARENT_FOLDER_ID = "1qZBLQ66_pwRwLOy3Zj5q_qAwY_Z05HXb"


def get_credentials():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    print(current_directory)
    credentital_file_path = os.path.join(current_directory, CREDENTIALS_FILE)
    print(credentital_file_path)

    creds = None

    # 既存のトークンを利用
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE) as token:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE)

    # 新規トークンを取得またはトークンの更新
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentital_file_path,
                ["https://www.googleapis.com/auth/drive.file"],
            )
            creds = flow.run_local_server(port=0)

        # トークンを保存
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())

    return creds


def upload_to_drive(filename, filepath, mimetype, folder_id=None):
    creds = get_credentials()
    service = build("drive", "v3", credentials=creds)

    file_metadata = {"name": filename, "parents": [folder_id]}
    media = MediaFileUpload(filepath, mimetype=mimetype)

    try:
        file = (
            service.files()
            .create(body=file_metadata, media_body=media, fields="id")
            .execute()
        )
        print(f'File ID: {file.get("id")}')
    except HttpError as error:
        print(f"An error occurred: {error}")


def create_folder(folder_name):
    creds = get_credentials()
    service = build("drive", "v3", credentials=creds)

    file_metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [PARENT_FOLDER_ID],
    }
    try:
        folder = service.files().create(body=file_metadata, fields="id").execute()
        return folder.get("id")
    except HttpError as error:
        print(f"An error occurred: {error}")
        return None

import os
import io
import re
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE")  # путь к файлу service account
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

# Авторизация
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)
drive_service = build('drive', 'v3', credentials=credentials)

def extract_folder_id_from_url(url: str) -> str:
    """
    Извлекает folder_id из ссылки Google Drive.
    Поддерживает ссылки вида:
    - https://drive.google.com/drive/folders/<folder_id>
    - https://drive.google.com/drive/folders/<folder_id>?usp=sharing
    """
    match = re.search(r'/folders/([a-zA-Z0-9-_]+)', url)
    if match:
        return match.group(1)
    else:
        raise ValueError("Не удалось извлечь folder_id из ссылки")

def list_files_in_folder(folder_id):
    """Возвращает список всех файлов и папок в папке."""
    query = f"'{folder_id}' in parents and trashed=false"
    results = []
    page_token = None
    while True:
        response = drive_service.files().list(
            q=query,
            spaces='drive',
            fields='nextPageToken, files(id, name, mimeType)',
            pageToken=page_token
        ).execute()
        results.extend(response.get('files', []))
        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break
    return results

def download_file(file_id, file_path):
    """Скачивает один файл по ID."""
    if os.path.exists(file_path):
        logger.info(f"{file_path} уже скачан, пропускаем.")
        return

    request = drive_service.files().get_media(fileId=file_id)
    fh = io.FileIO(file_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        if status:
            logger.info(f"{os.path.basename(file_path)}: {int(status.progress() * 100)}% скачано")
    logger.info(f"{os.path.basename(file_path)} скачан успешно!")

def download_images_recursively(folder_id, local_path):
    """Рекурсивно скачивает все изображения из папки и подпапок."""
    os.makedirs(local_path, exist_ok=True)
    items = list_files_in_folder(folder_id)

    for item in items:
        item_name = item['name']
        item_id = item['id']
        mime_type = item['mimeType']

        if mime_type == 'application/vnd.google-apps.folder':
            # Рекурсивно спускаемся, но все файлы сохраняем в один local_path
            download_images_recursively(item_id, local_path)
        else:
            ext = os.path.splitext(item_name)[1].lower()
            if ext in ALLOWED_EXTENSIONS:
                download_file(item_id, os.path.join(local_path, item_name))
            else:
                logger.info(f"{item_name} пропущен (не изображение)")

if __name__ == "__main__":
    download_dir = "data/photos/raw_uploads"
    os.makedirs(download_dir, exist_ok=True)

    admin_url = input("Вставьте ссылку на папку Google Drive: ")
    folder_id = extract_folder_id_from_url(admin_url)
    download_images_recursively(folder_id, download_dir)
    logger.info("Загрузка завершена.")

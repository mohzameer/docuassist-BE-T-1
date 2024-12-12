from typing import List, Tuple, Optional, Set
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import pickle
import os
import io
from app.core.config import get_settings
from app.models.document import DocumentResponse
import logging

logger = logging.getLogger(__name__)
settings = get_settings()

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Google Workspace MIME types and their export formats
GOOGLE_MIME_TYPES = {
    'application/vnd.google-apps.document': 'application/pdf',
    'application/vnd.google-apps.spreadsheet': 'application/pdf',
    'application/vnd.google-apps.presentation': 'application/pdf',
    'application/vnd.google-apps.drawing': 'application/pdf',
}

# Common document MIME types
DEFAULT_MIME_TYPES = [
    # Documents
    'application/pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/msword',
    'application/vnd.google-apps.document',
    'text/plain',
    'text/csv',
    'application/rtf',
    
    # Spreadsheets
    'application/vnd.google-apps.spreadsheet',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/vnd.ms-excel',
    
    # Presentations
    'application/vnd.google-apps.presentation',
    'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    'application/vnd.ms-powerpoint',
    
    # Images
    'image/jpeg',
    'image/png',
    'image/gif',
    'image/tiff',
    'image/bmp',
    'image/webp'
]

FOLDER_MIME_TYPE = 'application/vnd.google-apps.folder'

class GoogleDriveService:
    def __init__(self):
        self.creds = self._get_credentials()
        self.service = build('drive', 'v3', credentials=self.creds)

    def _get_credentials(self) -> Credentials:
        """Get valid credentials for Google Drive API"""
        creds = None
        if os.path.exists(settings.GOOGLE_DRIVE_TOKEN_FILE):
            with open(settings.GOOGLE_DRIVE_TOKEN_FILE, 'rb') as token:
                creds = pickle.load(token)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    settings.GOOGLE_DRIVE_CREDENTIALS_FILE, SCOPES)
                creds = flow.run_local_server(port=0)
            with open(settings.GOOGLE_DRIVE_TOKEN_FILE, 'wb') as token:
                pickle.dump(creds, token)

        return creds

    async def _list_folder_contents(self, folder_id: str, recursive: bool = False) -> List[str]:
        """List all file IDs in a folder, including subfolders if recursive"""
        try:
            query = f"'{folder_id}' in parents and mimeType != '{FOLDER_MIME_TYPE}'"
            results = self.service.files().list(
                q=query,
                fields="files(id, mimeType)",
                pageSize=1000
            ).execute()
            
            file_ids = [item['id'] for item in results.get('files', [])]
            
            if recursive:
                # Get subfolders
                folder_query = f"'{folder_id}' in parents and mimeType = '{FOLDER_MIME_TYPE}'"
                folder_results = self.service.files().list(
                    q=folder_query,
                    fields="files(id)",
                    pageSize=1000
                ).execute()
                
                # Recursively get files from subfolders
                for folder in folder_results.get('files', []):
                    subfolder_files = await self._list_folder_contents(folder['id'], recursive=True)
                    file_ids.extend(subfolder_files)
                    
            return file_ids
            
        except Exception as e:
            logger.error(f"Error listing folder contents: {e}", exc_info=True)
            raise

    async def _get_file_details(self, file_id: str) -> Optional[dict]:
        """Get detailed information for a single file"""
        try:
            return self.service.files().get(
                fileId=file_id,
                fields="id, name, mimeType, createdTime, modifiedTime, size, description"
            ).execute()
        except Exception as e:
            logger.error(f"Error getting file details for {file_id}: {e}")
            return None

    async def list_files(
        self,
        folder_id: Optional[str] = None,
        mime_types: Optional[List[str]] = None,
        recursive: bool = False
    ) -> List[DocumentResponse]:
        """List files from Google Drive"""
        try:
            # Get all file IDs from the folder and subfolders
            if folder_id or settings.GOOGLE_DRIVE_FOLDER_ID:
                target_folder = folder_id or settings.GOOGLE_DRIVE_FOLDER_ID
                file_ids = await self._list_folder_contents(target_folder, recursive)
                if not file_ids:
                    logger.warning(f"No files found in folder {target_folder}")
                    return []
            else:
                return []

            # Use default mime types if none specified
            mime_types_to_use = set(mime_types if mime_types else DEFAULT_MIME_TYPES)
            
            # Get details for each file and filter by mime type
            documents = []
            for file_id in file_ids:
                file = await self._get_file_details(file_id)
                if file and file['mimeType'] in mime_types_to_use:
                    doc = DocumentResponse(
                        id=file['id'],
                        title=file['name'],
                        drive_id=file['id'],
                        mime_type=file['mimeType'],
                        created_at=file['createdTime'],
                        updated_at=file['modifiedTime'],
                        metadata={
                            'size': file.get('size'),
                            'description': file.get('description'),
                            'is_google_workspace': file['mimeType'] in GOOGLE_MIME_TYPES,
                            'export_format': GOOGLE_MIME_TYPES.get(file['mimeType'])
                        }
                    )
                    documents.append(doc)
                    logger.info(f"Added document: {doc.title} ({doc.mime_type})")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error listing files from Google Drive: {e}", exc_info=True)
            raise

    async def download_file(self, file_id: str, original_mime_type: str = None) -> Tuple[bytes, str]:
        """Download a file from Google Drive"""
        try:
            # Get file metadata first
            file_metadata = self.service.files().get(
                fileId=file_id,
                fields='mimeType'
            ).execute()
            
            mime_type = file_metadata.get('mimeType')
            
            # Skip folders
            if mime_type == FOLDER_MIME_TYPE:
                logger.warning(f"Attempted to download a folder: {file_id}")
                raise ValueError("Cannot download a folder")
                
            file_content = io.BytesIO()

            # Use original_mime_type if provided
            if original_mime_type in GOOGLE_MIME_TYPES:
                logger.info(f"Exporting Google Workspace file {file_id} as {GOOGLE_MIME_TYPES[original_mime_type]}")
                # Export Google Workspace files
                export_mime_type = GOOGLE_MIME_TYPES[original_mime_type]
                request = self.service.files().export_media(
                    fileId=file_id,
                    mimeType=export_mime_type
                )
                downloader = MediaIoBaseDownload(file_content, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                return file_content.getvalue(), export_mime_type
            else:
                logger.info(f"Downloading regular file {file_id} ({mime_type})")
                # Download regular files
                request = self.service.files().get_media(fileId=file_id)
                downloader = MediaIoBaseDownload(file_content, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                return file_content.getvalue(), mime_type

        except Exception as e:
            logger.error(f"Error downloading file from Google Drive: {e}", exc_info=True)
            raise 
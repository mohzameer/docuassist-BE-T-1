from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from typing import List, Optional, Dict, Any
import os
import io
import pickle
import logging
from datetime import datetime
from app.core.config import get_settings
from app.models.document import DocumentCreate, DocumentResponse

logger = logging.getLogger(__name__)
settings = get_settings()

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

class GoogleDriveService:
    def __init__(self):
        self.credentials = self._get_credentials()
        self.service = build('drive', 'v3', credentials=self.credentials)
        
    def _get_credentials(self) -> Credentials:
        """Get or refresh Google Drive credentials"""
        credentials = None
        
        # Try to load existing token
        if os.path.exists(settings.GOOGLE_DRIVE_TOKEN_FILE):
            with open(settings.GOOGLE_DRIVE_TOKEN_FILE, 'rb') as token:
                credentials = pickle.load(token)
        
        # If credentials are expired or don't exist, refresh or create new ones
        if not credentials or not credentials.valid:
            if credentials and credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    settings.GOOGLE_DRIVE_CREDENTIALS_FILE, SCOPES)
                credentials = flow.run_local_server(port=0)
            
            # Save credentials for future use
            with open(settings.GOOGLE_DRIVE_TOKEN_FILE, 'wb') as token:
                pickle.dump(credentials, token)
        
        return credentials

    async def list_files(
        self,
        folder_id: Optional[str] = None,
        mime_types: Optional[List[str]] = None,
        recursive: bool = False
    ) -> List[DocumentResponse]:
        """List files in the specified Google Drive folder"""
        try:
            folder_id = folder_id or settings.GOOGLE_DRIVE_FOLDER_ID
            query_parts = [
                f"'{folder_id}' in parents",
                "trashed = false"
            ]
            
            if mime_types:
                mime_type_query = " or ".join([f"mimeType='{mt}'" for mt in mime_types])
                query_parts.append(f"({mime_type_query})")
            
            query = " and ".join(query_parts)
            
            fields = "files(id, name, mimeType, createdTime, modifiedTime, size, webViewLink, thumbnailLink, parents)"
            
            results = []
            page_token = None
            
            while True:
                response = self.service.files().list(
                    q=query,
                    spaces='drive',
                    fields=fields,
                    pageToken=page_token
                ).execute()
                
                for file in response.get('files', []):
                    doc_response = DocumentResponse(
                        id=file['id'],
                        title=file['name'],
                        mime_type=file['mimeType'],
                        drive_id=file['id'],
                        parent_folder_id=file.get('parents', [None])[0],
                        created_at=datetime.fromisoformat(file['createdTime'].replace('Z', '+00:00')),
                        updated_at=datetime.fromisoformat(file['modifiedTime'].replace('Z', '+00:00')),
                        size=file.get('size'),
                        web_view_link=file.get('webViewLink'),
                        thumbnail_link=file.get('thumbnailLink'),
                        metadata={}
                    )
                    results.append(doc_response)
                
                page_token = response.get('nextPageToken')
                if not page_token:
                    break
                
            if recursive:
                # Get subfolders
                folder_query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed = false"
                folders = self.service.files().list(
                    q=folder_query,
                    spaces='drive',
                    fields='files(id)'
                ).execute()
                
                # Recursively get files from subfolders
                for folder in folders.get('files', []):
                    subfolder_files = await self.list_files(
                        folder_id=folder['id'],
                        mime_types=mime_types,
                        recursive=True
                    )
                    results.extend(subfolder_files)
            
            return results
            
        except Exception as e:
            logger.error(f"Error listing files from Google Drive: {e}", exc_info=True)
            raise

    async def download_file(self, file_id: str) -> tuple[bytes, str]:
        """Download a file from Google Drive"""
        try:
            file_metadata = self.service.files().get(fileId=file_id).execute()
            request = self.service.files().get_media(fileId=file_id)
            
            file_handle = io.BytesIO()
            downloader = MediaIoBaseDownload(file_handle, request)
            
            done = False
            while not done:
                _, done = downloader.next_chunk()
            
            return file_handle.getvalue(), file_metadata.get('mimeType', '')
            
        except Exception as e:
            logger.error(f"Error downloading file from Google Drive: {e}", exc_info=True)
            raise

    async def get_file_metadata(self, file_id: str) -> Dict[str, Any]:
        """Get metadata for a specific file"""
        try:
            return self.service.files().get(
                fileId=file_id,
                fields='id, name, mimeType, createdTime, modifiedTime, size, webViewLink, thumbnailLink, parents'
            ).execute()
            
        except Exception as e:
            logger.error(f"Error getting file metadata from Google Drive: {e}", exc_info=True)
            raise 
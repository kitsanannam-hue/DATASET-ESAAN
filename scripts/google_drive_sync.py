#!/usr/bin/env python3
"""
Google Drive Sync for AudioCraft Dataset
Import audio files from Google Drive and export datasets/models for backup.
"""

import os
import json
import requests
from pathlib import Path
from typing import Optional, List, Dict
import io


class GoogleDriveSync:
    """Sync audio files with Google Drive."""
    
    SUPPORTED_AUDIO_EXTENSIONS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']
    
    def __init__(self):
        self.connection_settings = None
        self.access_token = None
    
    def _get_access_token(self) -> str:
        """Get access token from Replit connector."""
        if self.connection_settings and self.connection_settings.get('settings', {}).get('expires_at'):
            from datetime import datetime
            expires_at = self.connection_settings['settings']['expires_at']
            if datetime.fromisoformat(expires_at.replace('Z', '+00:00')) > datetime.now():
                return self.connection_settings['settings']['access_token']
        
        hostname = os.environ.get('REPLIT_CONNECTORS_HOSTNAME')
        repl_identity = os.environ.get('REPL_IDENTITY')
        web_repl_renewal = os.environ.get('WEB_REPL_RENEWAL')
        
        if repl_identity:
            x_replit_token = f'repl {repl_identity}'
        elif web_repl_renewal:
            x_replit_token = f'depl {web_repl_renewal}'
        else:
            raise ValueError('X_REPLIT_TOKEN not found for repl/depl')
        
        response = requests.get(
            f'https://{hostname}/api/v2/connection?include_secrets=true&connector_names=google-drive',
            headers={
                'Accept': 'application/json',
                'X_REPLIT_TOKEN': x_replit_token
            }
        )
        
        data = response.json()
        self.connection_settings = data.get('items', [{}])[0] if data.get('items') else {}
        
        access_token = (
            self.connection_settings.get('settings', {}).get('access_token') or
            self.connection_settings.get('settings', {}).get('oauth', {}).get('credentials', {}).get('access_token')
        )
        
        if not access_token:
            raise ValueError('Google Drive not connected. Please connect Google Drive first.')
        
        self.access_token = access_token
        return access_token
    
    def _make_request(self, url: str, method: str = 'GET', **kwargs) -> requests.Response:
        """Make authenticated request to Google Drive API."""
        token = self._get_access_token()
        headers = kwargs.pop('headers', {})
        headers['Authorization'] = f'Bearer {token}'
        
        return requests.request(method, url, headers=headers, **kwargs)
    
    def list_files(self, folder_id: str = 'root', audio_only: bool = True) -> List[Dict]:
        """List files in a Google Drive folder."""
        query = f"'{folder_id}' in parents and trashed = false"
        
        if audio_only:
            mime_queries = [
                "mimeType = 'audio/wav'",
                "mimeType = 'audio/mpeg'",
                "mimeType = 'audio/mp3'",
                "mimeType = 'audio/flac'",
                "mimeType = 'audio/ogg'",
                "mimeType = 'audio/x-m4a'",
                "mimeType = 'audio/aac'",
                "mimeType = 'audio/x-wav'",
            ]
            query += f" and ({' or '.join(mime_queries)})"
        
        url = 'https://www.googleapis.com/drive/v3/files'
        params = {
            'q': query,
            'fields': 'files(id, name, mimeType, size)',
            'pageSize': 100
        }
        
        response = self._make_request(url, params=params)
        data = response.json()
        
        return data.get('files', [])
    
    def list_folders(self, parent_id: str = 'root') -> List[Dict]:
        """List folders in Google Drive."""
        query = f"'{parent_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        
        url = 'https://www.googleapis.com/drive/v3/files'
        params = {
            'q': query,
            'fields': 'files(id, name)',
            'pageSize': 100
        }
        
        response = self._make_request(url, params=params)
        data = response.json()
        
        return data.get('files', [])
    
    def download_file(self, file_id: str, dest_path: Path) -> bool:
        """Download a file from Google Drive."""
        url = f'https://www.googleapis.com/drive/v3/files/{file_id}?alt=media'
        
        response = self._make_request(url)
        
        if response.status_code == 200:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, 'wb') as f:
                f.write(response.content)
            return True
        return False
    
    def upload_file(self, file_path: Path, folder_id: str = 'root', name: Optional[str] = None) -> Optional[str]:
        """Upload a file to Google Drive."""
        name = name or file_path.name
        
        metadata = {
            'name': name,
            'parents': [folder_id]
        }
        
        url = 'https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart'
        
        token = self._get_access_token()
        
        from email.mime.multipart import MIMEMultipart
        from email.mime.base import MIMEBase
        import email.encoders
        
        files = {
            'metadata': ('metadata', json.dumps(metadata), 'application/json'),
            'file': (name, open(file_path, 'rb'))
        }
        
        response = requests.post(
            url,
            headers={'Authorization': f'Bearer {token}'},
            files=files
        )
        
        if response.status_code == 200:
            return response.json().get('id')
        return None
    
    def create_folder(self, name: str, parent_id: str = 'root') -> Optional[str]:
        """Create a folder in Google Drive."""
        metadata = {
            'name': name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_id]
        }
        
        url = 'https://www.googleapis.com/drive/v3/files'
        
        response = self._make_request(
            url,
            method='POST',
            json=metadata,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            return response.json().get('id')
        return None
    
    def import_audio_files(
        self,
        folder_id: str = 'root',
        dest_dir: str = 'data/raw/gdrive',
        recursive: bool = False
    ) -> List[str]:
        """Import audio files from Google Drive folder."""
        dest_path = Path(dest_dir)
        dest_path.mkdir(parents=True, exist_ok=True)
        
        imported_files = []
        
        print(f"Scanning Google Drive folder...")
        files = self.list_files(folder_id, audio_only=True)
        
        print(f"Found {len(files)} audio files")
        
        for file_info in files:
            file_name = file_info['name']
            file_id = file_info['id']
            
            dest_file = dest_path / file_name
            
            print(f"Downloading: {file_name}")
            if self.download_file(file_id, dest_file):
                imported_files.append(str(dest_file))
                print(f"  Saved to: {dest_file}")
            else:
                print(f"  Failed to download: {file_name}")
        
        if recursive:
            folders = self.list_folders(folder_id)
            for folder in folders:
                subfolder_dest = dest_path / folder['name']
                subfolder_files = self.import_audio_files(
                    folder['id'],
                    str(subfolder_dest),
                    recursive=True
                )
                imported_files.extend(subfolder_files)
        
        return imported_files
    
    def export_dataset(
        self,
        folder_name: str = 'AudioCraft_Dataset_Backup',
        include_manifests: bool = True,
        include_audio: bool = False
    ) -> Dict:
        """Export dataset to Google Drive."""
        print(f"Creating backup folder: {folder_name}")
        folder_id = self.create_folder(folder_name)
        
        if not folder_id:
            print("Failed to create backup folder")
            return {'success': False, 'error': 'Failed to create folder'}
        
        uploaded_files = []
        
        if include_manifests:
            manifest_files = [
                'data/processed/musicgen_train.jsonl',
                'data/processed/audiogen_train.jsonl',
                'data/dataset_summary.json'
            ]
            
            for manifest in manifest_files:
                path = Path(manifest)
                if path.exists():
                    print(f"Uploading: {path.name}")
                    file_id = self.upload_file(path, folder_id)
                    if file_id:
                        uploaded_files.append(manifest)
        
        if include_audio:
            audio_dirs = [
                'data/processed/musicgen',
                'data/processed/audiogen'
            ]
            
            for audio_dir in audio_dirs:
                dir_path = Path(audio_dir)
                if dir_path.exists():
                    subfolder_id = self.create_folder(dir_path.name, folder_id)
                    if subfolder_id:
                        for audio_file in dir_path.rglob('*.wav'):
                            print(f"Uploading: {audio_file.name}")
                            file_id = self.upload_file(audio_file, subfolder_id)
                            if file_id:
                                uploaded_files.append(str(audio_file))
        
        return {
            'success': True,
            'folder_id': folder_id,
            'uploaded_files': uploaded_files
        }


def main():
    """Main function for Google Drive sync."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Google Drive Sync for AudioCraft')
    parser.add_argument('action', choices=['import', 'export', 'list'], help='Action to perform')
    parser.add_argument('--folder-id', default='root', help='Google Drive folder ID')
    parser.add_argument('--dest', default='data/raw/gdrive', help='Destination directory for imports')
    parser.add_argument('--recursive', action='store_true', help='Import recursively')
    parser.add_argument('--include-audio', action='store_true', help='Include audio files in export')
    
    args = parser.parse_args()
    
    sync = GoogleDriveSync()
    
    if args.action == 'list':
        print("=" * 60)
        print("Google Drive Contents")
        print("=" * 60)
        
        print("\nFolders:")
        folders = sync.list_folders(args.folder_id)
        for folder in folders:
            print(f"  [Folder] {folder['name']} (ID: {folder['id']})")
        
        print("\nAudio Files:")
        files = sync.list_files(args.folder_id, audio_only=True)
        for f in files:
            size_mb = int(f.get('size', 0)) / (1024 * 1024)
            print(f"  [Audio] {f['name']} ({size_mb:.2f} MB)")
        
        print(f"\nTotal: {len(folders)} folders, {len(files)} audio files")
    
    elif args.action == 'import':
        print("=" * 60)
        print("Importing Audio from Google Drive")
        print("=" * 60)
        
        imported = sync.import_audio_files(
            folder_id=args.folder_id,
            dest_dir=args.dest,
            recursive=args.recursive
        )
        
        print(f"\nImported {len(imported)} files")
        print("Run 'python scripts/generate_manifests.py' to update manifests")
    
    elif args.action == 'export':
        print("=" * 60)
        print("Exporting Dataset to Google Drive")
        print("=" * 60)
        
        result = sync.export_dataset(
            include_manifests=True,
            include_audio=args.include_audio
        )
        
        if result['success']:
            print(f"\nExport complete!")
            print(f"Folder ID: {result['folder_id']}")
            print(f"Uploaded {len(result['uploaded_files'])} files")
        else:
            print(f"\nExport failed: {result.get('error')}")


if __name__ == "__main__":
    main()

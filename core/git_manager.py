import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
import validators
import requests
import zipfile
import tempfile
from langchain.schema import Document
from utils.config import Config

# Disable SSL warnings globally
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class GitManager:
    """Manages Git repository operations using ZIP download (SSL-friendly)"""
    
    def __init__(self):
        self.repositories_path = Path(Config.REPOSITORIES_PATH)
        self.repositories_path.mkdir(parents=True, exist_ok=True)
        
        # Configure requests session with SSL disabled
        self.session = requests.Session()
        self.session.verify = False
        self.session.timeout = 120
    
    def validate_git_url(self, url: str) -> bool:
        """Validate if URL is a valid Git repository URL"""
        if not validators.url(url):
            return False
        
        # Check if it's a GitHub/GitLab URL
        valid_domains = ['github.com', 'gitlab.com', 'bitbucket.org']
        return any(domain in url.lower() for domain in valid_domains)
    
    def get_repo_name_from_url(self, url: str) -> str:
        """Extract repository name from URL"""
        # Remove .git extension and get the last part
        repo_name = url.rstrip('/').split('/')[-1]
        if repo_name.endswith('.git'):
            repo_name = repo_name[:-4]
        
        # Add timestamp if repository already exists to avoid conflicts
        repo_path = self.repositories_path / repo_name
        if repo_path.exists():
            import time
            timestamp = int(time.time())
            repo_name = f"{repo_name}_{timestamp}"
        
        return repo_name
    
    def clone_and_load_repository(self, url: str, progress_callback=None) -> Dict[str, Any]:
        """Download repository as ZIP and load documents"""
        try:
            if not self.validate_git_url(url):
                return {
                    'success': False,
                    'error': f'Invalid Git URL: {url}',
                    'documents': [],
                    'repo_info': {}
                }
            
            repo_name = self.get_repo_name_from_url(url)
            repo_path = self.repositories_path / repo_name
            
            # Remove existing directory if it exists
            if repo_path.exists():
                if progress_callback:
                    progress_callback(f"Removing existing {repo_name}...")
                self._force_remove_directory(repo_path)
            
            if progress_callback:
                progress_callback(f"Downloading {repo_name} repository...")
            
            # Download repository as ZIP
            documents = self._download_and_process_repository(url, repo_path, repo_name, progress_callback)
            
            if not documents:
                return {
                    'success': False,
                    'error': 'No documents found in repository or download failed',
                    'documents': [],
                    'repo_info': {}
                }
            
            if progress_callback:
                progress_callback(f"Loaded {len(documents)} documents from {repo_name}")
            
            # Get repository info
            repo_info = self._get_repository_info(repo_path, documents)
            
            return {
                'success': True,
                'error': None,
                'documents': documents,
                'repo_info': repo_info,
                'repo_name': repo_name,
                'repo_path': str(repo_path),
                'url': url
            }
            
        except Exception as e:
            error_msg = f'Failed to download repository: {str(e)}'
            if progress_callback:
                progress_callback(f"Error: {error_msg}")
            
            return {
                'success': False,
                'error': error_msg,
                'documents': [],
                'repo_info': {}
            }
    
    def _download_and_process_repository(self, url: str, repo_path: Path, repo_name: str, progress_callback=None) -> List[Document]:
        """Download repository as ZIP and process files"""
        try:
            if progress_callback:
                progress_callback(f"Downloading {repo_name} ZIP archive...")
            
            # Convert to ZIP download URL
            if 'github.com' in url:
                # Extract owner and repo from URL
                parts = url.rstrip('/').split('/')
                if len(parts) >= 2:
                    owner = parts[-2]
                    repo = parts[-1]
                    
                    # Try main branch first, then master
                    zip_urls = [
                        f"https://github.com/{owner}/{repo}/archive/refs/heads/main.zip",
                        f"https://github.com/{owner}/{repo}/archive/refs/heads/master.zip"
                    ]
                else:
                    raise ValueError("Invalid GitHub URL format")
            else:
                raise ValueError("Only GitHub repositories are supported in this version")
            
            # Try downloading
            downloaded = False
            for zip_url in zip_urls:
                try:
                    if progress_callback:
                        progress_callback(f"Trying download from {zip_url.split('/')[-1]}...")
                    
                    response = self.session.get(zip_url, stream=True)
                    response.raise_for_status()
                    
                    # Download and extract
                    self._extract_zip_to_directory(response, repo_path, progress_callback)
                    downloaded = True
                    break
                    
                except requests.exceptions.RequestException as e:
                    if progress_callback:
                        progress_callback(f"Failed with {zip_url.split('/')[-1]}: {str(e)}")
                    continue
            
            if not downloaded:
                raise Exception("Failed to download from both main and master branches")
            
            if progress_callback:
                progress_callback(f"Processing files from {repo_name}...")
            
            # Process downloaded files
            documents = self._process_files_from_directory(repo_path)
            return documents
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"Download failed: {str(e)}")
            raise
    
    def _extract_zip_to_directory(self, response, repo_path: Path, progress_callback=None):
        """Extract ZIP response to directory"""
        # Save ZIP to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            tmp_file_path = tmp_file.name
        
        try:
            # Remove existing directory if it exists (with force)
            if repo_path.exists():
                self._force_remove_directory(repo_path)
            
            # Extract ZIP
            with zipfile.ZipFile(tmp_file_path, 'r') as zip_ref:
                zip_ref.extractall(self.repositories_path)
            
            # Find extracted folder and rename it
            extracted_folders = [f for f in self.repositories_path.iterdir() 
                               if f.is_dir() and repo_path.name in f.name]
            
            if extracted_folders:
                extracted_folder = extracted_folders[0]
                if extracted_folder != repo_path:
                    # Force remove target if it exists
                    if repo_path.exists():
                        self._force_remove_directory(repo_path)
                    
                    # Rename extracted folder
                    extracted_folder.rename(repo_path)
                    
            if progress_callback:
                progress_callback(f"Successfully extracted to {repo_path.name}")
                
        finally:
            # Cleanup temporary file
            if os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass  # Ignore cleanup errors
    
    def _force_remove_directory(self, directory_path: Path):
        """Force remove directory even if files are locked"""
        if not directory_path.exists():
            return
        
        try:
            # Try normal removal first
            shutil.rmtree(directory_path)
        except (PermissionError, OSError) as e:
            # Try harder removal on Windows
            import stat
            import time
            
            def handle_remove_readonly(func, path, exc):
                """Handle readonly files during removal"""
                try:
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
                except:
                    pass
            
            # Wait a bit and try again
            time.sleep(0.5)
            try:
                shutil.rmtree(directory_path, onerror=handle_remove_readonly)
            except:
                # Last resort - try individual file removal
                try:
                    for root, dirs, files in os.walk(directory_path, topdown=False):
                        for file in files:
                            try:
                                file_path = os.path.join(root, file)
                                os.chmod(file_path, stat.S_IWRITE)
                                os.remove(file_path)
                            except:
                                pass
                        for dir in dirs:
                            try:
                                os.rmdir(os.path.join(root, dir))
                            except:
                                pass
                    os.rmdir(directory_path)
                except:
                    pass  # Give up if nothing works
    
    def _process_files_from_directory(self, repo_path: Path) -> List[Document]:
        """Process files from directory into Document objects"""
        documents = []
        
        for file_path in repo_path.rglob('*'):
            if file_path.is_file() and self._should_process_file(file_path, repo_path):
                try:
                    # Try different encodings
                    content = None
                    for encoding in ['utf-8', 'latin-1', 'cp1252']:
                        try:
                            content = file_path.read_text(encoding=encoding)
                            break
                        except UnicodeDecodeError:
                            continue
                    
                    if content is None:
                        continue  # Skip files that can't be decoded
                    
                    # Create document with metadata
                    relative_path = str(file_path.relative_to(repo_path))
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            'file_path': relative_path,
                            'file_name': file_path.name,
                            'source': str(file_path),
                            'file_size': len(content)
                        }
                    )
                    documents.append(doc)
                    
                except Exception as e:
                    continue  # Skip files that can't be processed
        
        return documents
    
    def _should_process_file(self, file_path: Path, repo_path: Path) -> bool:
        """Check if file should be processed"""
        # Check extension
        if file_path.suffix.lower() not in Config.SUPPORTED_EXTENSIONS:
            return False
        
        # Check excluded directories
        relative_path = file_path.relative_to(repo_path)
        path_parts = set(relative_path.parts)
        if path_parts.intersection(Config.EXCLUDE_DIRS):
            return False
        
        # Check file size
        try:
            if file_path.stat().st_size > Config.MAX_FILE_SIZE:
                return False
            if file_path.stat().st_size == 0:  # Skip empty files
                return False
        except:
            return False
        
        return True
    
    def _get_repository_info(self, repo_path: Path, documents: List[Document]) -> Dict[str, Any]:
        """Get information about the loaded repository"""
        try:
            # Count files by language/extension
            language_counts = {}
            file_counts = {}
            total_size = 0
            
            for doc in documents:
                # Extract info from document metadata
                file_path = doc.metadata.get('file_path', '')
                if file_path:
                    path = Path(file_path)
                    ext = path.suffix.lower()
                    file_counts[ext] = file_counts.get(ext, 0) + 1
                    
                    # Try to determine language
                    lang = self._get_language_from_extension(ext)
                    language_counts[lang] = language_counts.get(lang, 0) + 1
                    
                    # Add content size
                    total_size += len(doc.page_content)
            
            return {
                'name': repo_path.name,
                'path': str(repo_path),
                'total_documents': len(documents),
                'total_files': len(set(doc.metadata.get('file_path', '') for doc in documents)),
                'file_counts': file_counts,
                'language_counts': language_counts,
                'total_content_size': total_size,
                'size_mb': total_size / (1024 * 1024),
                'exists': True
            }
            
        except Exception as e:
            return {
                'exists': False,
                'error': str(e)
            }
    
    def _get_language_from_extension(self, ext: str) -> str:
        """Map file extension to programming language"""
        ext_to_lang = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.java': 'Java',
            '.go': 'Go',
            '.rs': 'Rust',
            '.cpp': 'C++',
            '.c': 'C',
            '.h': 'C/C++',
            '.json': 'JSON',
            '.yaml': 'YAML',
            '.yml': 'YAML',
        }
        return ext_to_lang.get(ext.lower(), 'Other')
    
    def list_repositories(self) -> Dict[str, Dict[str, Any]]:
        """List all downloaded repositories"""
        repositories = {}
        
        for repo_dir in self.repositories_path.iterdir():
            if repo_dir.is_dir():
                repo_info = {
                    'name': repo_dir.name,
                    'path': str(repo_dir),
                    'exists': True
                }
                repositories[repo_dir.name] = repo_info
        
        return repositories
    
    def remove_repository(self, repo_name: str) -> bool:
        """Remove a downloaded repository"""
        try:
            repo_path = self.repositories_path / repo_name
            if repo_path.exists():
                self._force_remove_directory(repo_path)
                return True
            return False
        except Exception:
            return False
import streamlit as st
from typing import Dict, Any, List
from core.git_manager import GitManager
from core.document_processor import DocumentProcessor
from core.vector_store import VectorStoreManager
from utils.config import Config

class StartupProcessor:
    """Handles application startup and auto-processing of repositories"""
    
    def __init__(self):
        self.git_manager = GitManager()
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStoreManager()
    
    def initialize_app(self) -> Dict[str, Any]:
        """Initialize app with pre-configured repositories"""
        
        # Check if vector store already has data
        stats = self.vector_store.get_collection_stats()
        existing_repos = stats.get('repositories', [])
        
        if existing_repos and stats.get('total_documents', 0) > 0:
            # Convert existing repo list to proper format
            formatted_repos = []
            for repo_name in existing_repos:
                formatted_repos.append({
                    'name': repo_name,
                    'documents': 'N/A',
                    'files': 'N/A'
                })
            
            return {
                'success': True,
                'message': f"App ready! {stats.get('total_documents', 0)} documents from {len(existing_repos)} repositories.",
                'repositories': formatted_repos,
                'stats': stats,
                'already_initialized': True
            }
        
        # Process default repositories
        return self._process_default_repositories()
    
    def _process_default_repositories(self) -> Dict[str, Any]:
        """Process all default repositories on startup"""
        
        if not Config.DEFAULT_REPOSITORIES:
            return {
                'success': True,
                'message': "No default repositories configured.",
                'repositories': [],
                'stats': {},
                'already_initialized': False
            }
        
        processed_repos = []
        total_documents = 0
        
        # Create progress placeholders
        progress_container = st.container()
        
        with progress_container:
            st.info("🚀 Initializing AI Code Analyzer...")
            
            for i, repo_url in enumerate(Config.DEFAULT_REPOSITORIES):
                repo_name = self.git_manager.get_repo_name_from_url(repo_url)
                
                progress_placeholder = st.empty()
                
                def progress_callback(message):
                    progress_placeholder.info(f"[{i+1}/{len(Config.DEFAULT_REPOSITORIES)}] {message}")
                
                try:
                    # Download and load repository
                    progress_callback(f"Loading {repo_name}...")
                    result = self.git_manager.clone_and_load_repository(repo_url, progress_callback)
                    
                    if not result['success']:
                        progress_placeholder.error(f"Failed to load {repo_name}: {result['error']}")
                        continue
                    
                    # Process documents
                    progress_callback(f"Processing {repo_name} for AI analysis...")
                    documents = result['documents']
                    processed_docs = self.document_processor.process_documents(
                        documents, 
                        repo_name, 
                        progress_callback
                    )
                    
                    # Add to vector store
                    progress_callback(f"Adding {repo_name} to vector database...")
                    vector_result = self.vector_store.add_documents(processed_docs, progress_callback)
                    
                    if vector_result['success']:
                        processed_repos.append({
                            'name': repo_name,
                            'url': repo_url,
                            'documents': len(processed_docs),
                            'files': result['repo_info'].get('total_files', 0)
                        })
                        total_documents += len(processed_docs)
                        progress_placeholder.success(f"✅ {repo_name} ready! ({len(processed_docs)} chunks)")
                    else:
                        progress_placeholder.error(f"Failed to process {repo_name}: {vector_result.get('error', 'Unknown error')}")
                
                except Exception as e:
                    progress_placeholder.error(f"Error with {repo_name}: {str(e)}")
                    continue
            
            # Final status
            if processed_repos:
                st.success(f"🎉 Initialization complete! {total_documents} documents from {len(processed_repos)} repositories ready for analysis.")
            else:
                st.error("❌ No repositories were successfully processed.")
        
        return {
            'success': len(processed_repos) > 0,
            'message': f"Processed {len(processed_repos)} repositories with {total_documents} documents.",
            'repositories': processed_repos,
            'stats': self.vector_store.get_collection_stats(),
            'already_initialized': False
        }
    
    def add_additional_repository(self, repo_url: str, progress_callback=None) -> Dict[str, Any]:
        """Add an additional repository to existing setup"""
        try:
            repo_name = self.git_manager.get_repo_name_from_url(repo_url)
            
            # Check if repository already exists
            existing_stats = self.vector_store.get_collection_stats()
            if repo_name in existing_stats.get('repositories', []):
                return {
                    'success': False,
                    'error': f"Repository '{repo_name}' is already loaded."
                }
            
            if progress_callback:
                progress_callback(f"Loading {repo_name}...")
            
            # Download and load repository
            result = self.git_manager.clone_and_load_repository(repo_url, progress_callback)
            
            if not result['success']:
                return {
                    'success': False,
                    'error': f"Failed to load repository: {result['error']}"
                }
            
            # Process documents
            if progress_callback:
                progress_callback(f"Processing {repo_name}...")
            
            documents = result['documents']
            processed_docs = self.document_processor.process_documents(
                documents, 
                repo_name, 
                progress_callback
            )
            
            # Add to vector store
            if progress_callback:
                progress_callback(f"Adding to vector database...")
            
            vector_result = self.vector_store.add_documents(processed_docs, progress_callback)
            
            if vector_result['success']:
                return {
                    'success': True,
                    'repo_name': repo_name,
                    'documents_added': len(processed_docs),
                    'total_documents': vector_result.get('total_documents', 0)
                }
            else:
                return {
                    'success': False,
                    'error': f"Failed to add to vector store: {vector_result.get('error', 'Unknown error')}"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Error processing repository: {str(e)}"
            }
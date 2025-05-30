from typing import List, Dict, Any
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.config import Config

class DocumentProcessor:
    """Processes documents for RAG (simplified since GitLoader handles file loading)"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def process_documents(self, documents: List[Document], repo_name: str, progress_callback=None) -> List[Document]:
        """Process documents by chunking and adding metadata"""
        processed_documents = []
        
        if progress_callback:
            progress_callback(f"Processing {len(documents)} files from {repo_name}...")
        
        for i, doc in enumerate(documents):
            if progress_callback and i % 10 == 0:
                progress_callback(f"Processing file {i+1}/{len(documents)}")
            
            # Split document into chunks
            chunks = self.text_splitter.split_text(doc.page_content)
            
            # Create new documents for each chunk
            for j, chunk in enumerate(chunks):
                # Enhance metadata
                enhanced_metadata = doc.metadata.copy()
                enhanced_metadata.update({
                    'repository': repo_name,
                    'chunk_index': j,
                    'total_chunks': len(chunks),
                    'chunk_id': f"{repo_name}_{enhanced_metadata.get('file_name', 'unknown')}_{j}",
                    'file_name': enhanced_metadata.get('file_path', '').split('/')[-1] if enhanced_metadata.get('file_path') else 'unknown',
                    'relative_path': enhanced_metadata.get('file_path', 'unknown'),
                    'language': self._detect_language_from_path(enhanced_metadata.get('file_path', '')),
                    'chunk_size': len(chunk)
                })
                
                processed_doc = Document(
                    page_content=chunk,
                    metadata=enhanced_metadata
                )
                processed_documents.append(processed_doc)
        
        if progress_callback:
            progress_callback(f"Created {len(processed_documents)} document chunks from {len(documents)} files")
        
        return processed_documents
    
    def _detect_language_from_path(self, file_path: str) -> str:
        """Detect programming language from file path"""
        if not file_path:
            return 'Unknown'
        
        ext = file_path.split('.')[-1].lower() if '.' in file_path else ''
        
        ext_to_lang = {
            'py': 'Python',
            'js': 'JavaScript',
            'ts': 'TypeScript',
            'java': 'Java',
            'go': 'Go',
            'rs': 'Rust',
            'cpp': 'C++',
            'c': 'C',
            'h': 'C/C++',
            'json': 'JSON',
            'yaml': 'YAML',
            'yml': 'YAML',
        }
        
        return ext_to_lang.get(ext, 'Other')
    
    def get_processing_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """Get statistics about processed documents"""
        if not documents:
            return {}
        
        # Count by language
        language_counts = {}
        repo_counts = {}
        file_counts = {}
        
        for doc in documents:
            lang = doc.metadata.get('language', 'Unknown')
            repo = doc.metadata.get('repository', 'Unknown')
            file_name = doc.metadata.get('file_name', 'Unknown')
            
            language_counts[lang] = language_counts.get(lang, 0) + 1
            repo_counts[repo] = repo_counts.get(repo, 0) + 1
            file_counts[file_name] = file_counts.get(file_name, 0) + 1
        
        return {
            'total_documents': len(documents),
            'language_distribution': language_counts,
            'repository_distribution': repo_counts,
            'unique_files': len(file_counts),
            'avg_chunk_size': sum(len(doc.page_content) for doc in documents) / len(documents) if documents else 0
        }
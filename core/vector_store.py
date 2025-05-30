import os
from typing import List, Dict, Any, Optional
import chromadb
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from utils.config import Config

class VectorStoreManager:
    """Manages ChromaDB vector store operations"""
    
    def __init__(self):
        self.embeddings = self._initialize_embeddings()
        self.vector_store = None
        self.collection_name = "code_repository"
        self._initialize_vector_store()
    
    def _initialize_embeddings(self) -> AzureOpenAIEmbeddings:
        """Initialize Azure OpenAI embeddings"""
        return AzureOpenAIEmbeddings(
            azure_deployment=Config.EMBEDDING_DEPLOYMENT_NAME,
            api_version=Config.OPENAI_API_VERSION,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            api_key=Config.AZURE_OPENAI_API_KEY
        )
    
    def _initialize_vector_store(self):
        """Initialize ChromaDB vector store"""
        try:
            # Initialize Chroma vector store with automatic persistence
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=Config.CHROMADB_PERSIST_DIRECTORY
            )
            
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise
    
    def add_documents(self, documents: List[Document], progress_callback=None) -> Dict[str, Any]:
        """Add documents to the vector store"""
        try:
            if not documents:
                return {'success': False, 'error': 'No documents provided'}
            
            if progress_callback:
                progress_callback(f"Adding {len(documents)} documents to vector store...")
            
            # Add documents in batches to avoid memory issues
            batch_size = 100
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                
                if progress_callback:
                    progress_callback(f"Processing batch {batch_num}/{total_batches}")
                
                # Add batch to vector store
                self.vector_store.add_documents(batch)
            
            # Note: persist() is no longer needed in langchain-chroma
            # The data is automatically persisted
            
            if progress_callback:
                progress_callback("Documents successfully added to vector store")
            
            return {
                'success': True,
                'documents_added': len(documents),
                'total_documents': self.get_document_count()
            }
            
        except Exception as e:
            error_msg = f"Error adding documents to vector store: {e}"
            if progress_callback:
                progress_callback(error_msg)
            return {'success': False, 'error': error_msg}
    
    def similarity_search(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[Document]:
        """Search for similar documents"""
        try:
            if filter_dict:
                # Use metadata filtering if provided
                results = self.vector_store.similarity_search(query, k=k, filter=filter_dict)
            else:
                results = self.vector_store.similarity_search(query, k=k)
            
            return results
            
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []
    
    def similarity_search_with_scores(self, query: str, k: int = 5) -> List[tuple]:
        """Search for similar documents with relevance scores"""
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            print(f"Error in similarity search with scores: {e}")
            return []
    
    def get_document_count(self) -> int:
        """Get total number of documents in vector store"""
        try:
            # Access the underlying ChromaDB collection
            client = chromadb.PersistentClient(path=Config.CHROMADB_PERSIST_DIRECTORY)
            collection = client.get_collection(name=self.collection_name)
            return collection.count()
        except Exception as e:
            print(f"Error getting document count: {e}")
            return 0
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection"""
        try:
            client = chromadb.PersistentClient(path=Config.CHROMADB_PERSIST_DIRECTORY)
            collection = client.get_collection(name=self.collection_name)
            
            # Get all metadata to analyze
            results = collection.get(include=['metadatas'])
            
            if not results['metadatas']:
                return {'total_documents': 0}
            
            # Analyze metadata
            repositories = set()
            languages = {}
            file_types = {}
            
            for metadata in results['metadatas']:
                if metadata:
                    repo = metadata.get('repository', 'Unknown')
                    lang = metadata.get('language', 'Unknown')
                    ext = metadata.get('file_extension', 'Unknown')
                    
                    repositories.add(repo)
                    languages[lang] = languages.get(lang, 0) + 1
                    file_types[ext] = file_types.get(ext, 0) + 1
            
            return {
                'total_documents': len(results['metadatas']),
                'repositories': list(repositories),
                'repository_count': len(repositories),
                'language_distribution': languages,
                'file_type_distribution': file_types
            }
            
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {}
    
    def delete_repository_documents(self, repository_name: str) -> bool:
        """Delete all documents from a specific repository"""
        try:
            client = chromadb.PersistentClient(path=Config.CHROMADB_PERSIST_DIRECTORY)
            collection = client.get_collection(name=self.collection_name)
            
            # Get all documents from the repository
            results = collection.get(
                where={"repository": repository_name},
                include=['ids']
            )
            
            if results['ids']:
                collection.delete(ids=results['ids'])
                return True
            
            return False
            
        except Exception as e:
            print(f"Error deleting repository documents: {e}")
            return False
    
    def clear_vector_store(self) -> bool:
        """Clear all documents from vector store"""
        try:
            # Get client and delete collection
            client = chromadb.PersistentClient(path=Config.CHROMADB_PERSIST_DIRECTORY)
            
            # Try to delete the collection
            try:
                client.delete_collection(name=self.collection_name)
            except Exception:
                pass  # Collection might not exist
            
            # Reinitialize the vector store
            self._initialize_vector_store()
            return True
            
        except Exception as e:
            print(f"Error clearing vector store: {e}")
            return False
    
    def search_by_repository(self, query: str, repository_name: str, k: int = 5) -> List[Document]:
        """Search within a specific repository"""
        filter_dict = {"repository": repository_name}
        return self.similarity_search(query, k=k, filter_dict=filter_dict)
    
    def get_retriever(self, search_kwargs: Optional[Dict] = None):
        """Get a retriever for the vector store"""
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
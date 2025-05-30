from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain_openai import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from core.vector_store import VectorStoreManager
from utils.config import Config

class RAGChain:
    """RAG Chain for code analysis and Q&A"""
    
    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store_manager = vector_store_manager
        self.llm = self._initialize_llm()
        self.chat_history = []  # Simple list to store chat history
        self.chain = self._create_chain()
    
    def _initialize_llm(self) -> AzureChatOpenAI:
        """Initialize Azure OpenAI LLM"""
        return AzureChatOpenAI(
            azure_deployment=Config.CHAT_DEPLOYMENT_NAME,
            api_version=Config.OPENAI_API_VERSION,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            api_key=Config.AZURE_OPENAI_API_KEY,
            temperature=0.1,  # Low temperature for consistent code analysis
            max_tokens=2000
        )
    
    def _create_code_analysis_prompt(self) -> PromptTemplate:
        """Create specialized prompt for code analysis"""
        template = """You are an expert code analyst and software architect. Your job is to analyze codebases and answer questions about software architecture, code structure, and implementation details.

Context from the codebase:
{context}

Chat History:
{chat_history}

Instructions:
1. Analyze the provided code context carefully
2. Answer questions about:
   - Software architecture and design patterns
   - Data flow between components
   - Control flow and function call chains
   - Purpose and functionality of code elements
   - Code relationships and dependencies
3. Always provide specific file names and locations when referencing code
4. Structure your answers clearly with bullet points or numbered lists when appropriate
5. If you're unsure about something, say so rather than guessing
6. Focus on practical insights that would help developers understand the codebase

Human Question: {question}

Detailed Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "chat_history", "question"]
        )
    
    def _create_chain(self) -> ConversationalRetrievalChain:
        """Create the conversational retrieval chain"""
        # Get retriever from vector store
        retriever = self.vector_store_manager.get_retriever(
            search_kwargs={"k": 8}  # Retrieve more chunks for comprehensive analysis
        )
        
        # Create the chain without deprecated memory
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=True,
            verbose=False
        )
        
        return chain
    
    def _format_chat_history(self) -> str:
        """Format chat history as string for prompt"""
        if not self.chat_history:
            return "No previous conversation."
        
        formatted_history = []
        for item in self.chat_history[-3:]:  # Last 3 exchanges
            formatted_history.append(f"Human: {item['question']}")
            formatted_history.append(f"Assistant: {item['answer'][:200]}...")  # Truncate for brevity
        
        return "\n".join(formatted_history)
    
    def ask_question(self, question: str, repository_filter: Optional[str] = None) -> Dict[str, Any]:
        """Ask a question about the codebase"""
        try:
            # Get retriever
            if repository_filter:
                retriever = self.vector_store_manager.vector_store.as_retriever(
                    search_kwargs={
                        "k": 8,
                        "filter": {"repository": repository_filter}
                    }
                )
            else:
                retriever = self.vector_store_manager.get_retriever(search_kwargs={"k": 8})
            
            # Get relevant documents
            relevant_docs = retriever.invoke(question)
            
            # Create context from documents
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Create chat history string
            chat_history_str = ""
            if self.chat_history:
                chat_history_str = "\n".join([
                    f"Previous Q: {item['question']}\nPrevious A: {item['answer'][:100]}..."
                    for item in self.chat_history[-2:]  # Last 2 exchanges
                ])
            
            # Create prompt
            prompt = f"""You are an expert code analyst. Analyze the provided code context and answer the question.

Context from codebase:
{context}

Recent conversation:
{chat_history_str}

Question: {question}

Provide a detailed, accurate answer with specific file references when possible."""
            
            # Get response from LLM
            response = self.llm.invoke(prompt)
            
            # Process source documents
            source_info = self._process_source_documents(relevant_docs)
            
            # Store in chat history
            self.chat_history.append({
                "question": question,
                "answer": response.content
            })
            
            # Keep only last 5 exchanges
            if len(self.chat_history) > 5:
                self.chat_history = self.chat_history[-5:]
            
            return {
                "answer": response.content,
                "source_documents": source_info,
                "question": question,
                "success": True
            }
            
        except Exception as e:
            return {
                "answer": f"Error processing question: {str(e)}",
                "source_documents": [],
                "question": question,
                "success": False,
                "error": str(e)
            }
    
    def _process_source_documents(self, source_docs: List[Document]) -> List[Dict[str, Any]]:
        """Process source documents for display"""
        processed_sources = []
        
        for doc in source_docs:
            metadata = doc.metadata
            processed_sources.append({
                "file_path": metadata.get("file_path", "Unknown"),
                "repository": metadata.get("repository", "Unknown"),
                "language": metadata.get("language", "Unknown"),
                "file_name": metadata.get("file_name", "Unknown"),
                "chunk_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "chunk_id": metadata.get("chunk_id", "Unknown")
            })
        
        return processed_sources
    
    def get_repository_summary(self, repository_name: str) -> Dict[str, Any]:
        """Get a summary of a specific repository"""
        try:
            # Search for general repository information
            docs = self.vector_store_manager.search_by_repository(
                "main functions classes modules components architecture",
                repository_name,
                k=10
            )
            
            if not docs:
                return {
                    "summary": f"No documents found for repository: {repository_name}",
                    "success": False
                }
            
            # Create context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Generate summary
            summary_prompt = f"""Analyze this codebase and provide a comprehensive summary:

{context}

Please provide:
1. Main purpose and functionality of this repository
2. Key components and modules
3. Programming languages and frameworks used
4. Architecture overview
5. Notable patterns or design approaches

Summary:"""
            
            response = self.llm.invoke(summary_prompt)
            
            return {
                "summary": response.content,
                "repository": repository_name,
                "analyzed_files": len(set(doc.metadata.get("file_name", "") for doc in docs)),
                "success": True
            }
            
        except Exception as e:
            return {
                "summary": f"Error generating repository summary: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.chat_history = []
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.chat_history.copy()
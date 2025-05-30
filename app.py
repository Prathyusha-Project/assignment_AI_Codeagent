import streamlit as st
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.startup_processor import StartupProcessor
from core.rag_chain import RAGChain
from utils.config import Config

# Page configuration
st.set_page_config(
    page_title="AI Code Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state"""
    if 'app_initialized' not in st.session_state:
        st.session_state.app_initialized = False
    
    if 'startup_processor' not in st.session_state:
        st.session_state.startup_processor = StartupProcessor()
    
    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain = RAGChain(st.session_state.startup_processor.vector_store)
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'repositories_info' not in st.session_state:
        st.session_state.repositories_info = []

def startup_initialization():
    """Handle app startup and initialization"""
    if st.session_state.app_initialized:
        return
    
    st.title("🔍 AI Code Analyzer")
    st.markdown("*Initializing your AI-powered code analysis environment...*")
    
    try:
        # Initialize app with default repositories
        with st.spinner("Setting up your code analysis environment..."):
            result = st.session_state.startup_processor.initialize_app()
        
        if result['success']:
            st.session_state.app_initialized = True
            st.session_state.repositories_info = result.get('repositories', [])
            
            if result.get('already_initialized'):
                st.success("🎉 " + result['message'])
            else:
                st.success("✅ " + result['message'])
            
            # Show initialization results
            repositories = result.get('repositories', [])
            if repositories:
                st.markdown("### 📚 Loaded Repositories:")
                for repo in repositories:
                    # Handle both string and dict formats safely
                    try:
                        if isinstance(repo, str):
                            st.markdown(f"- **{repo}**: Ready for analysis")
                        elif isinstance(repo, dict):
                            repo_name = repo.get('name', 'Unknown')
                            documents = repo.get('documents', 'N/A')
                            files = repo.get('files', 'N/A')
                            st.markdown(f"- **{repo_name}**: {documents} document chunks from {files} files")
                        else:
                            st.markdown(f"- Repository loaded successfully")
                    except Exception as e:
                        st.markdown(f"- Repository loaded (display error: {str(e)})")
            
            st.markdown("---")
            st.info("🚀 **Ready!** You can now ask questions about your codebase below.")
            
            # Small delay to show success message
            import time
            time.sleep(2)
            st.rerun()
        else:
            st.error("❌ Failed to initialize application. Please check your configuration and try again.")
            st.error(f"Error details: {result.get('message', 'Unknown error')}")
            st.stop()
            
    except Exception as e:
        st.error(f"❌ Startup error: {str(e)}")
        st.error("Please check your configuration and try again.")
        st.stop()

def sidebar_management():
    """Sidebar for additional repository management"""
    with st.sidebar:
        st.header("📊 Repository Status")
        
        # Show current stats (force refresh)
        if st.session_state.app_initialized:
            # Get fresh stats from vector store
            stats = st.session_state.startup_processor.vector_store.get_collection_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", stats.get('total_documents', 0))
            with col2:
                st.metric("Repositories", stats.get('repository_count', 0))
            
            # Show repository list
            st.subheader("📚 Current Repositories")
            repositories = stats.get('repositories', [])
            if repositories:
                for repo_name in repositories:
                    st.write(f"✅ {repo_name}")
            else:
                st.write("No repositories loaded")
        
        st.markdown("---")
        
        # Add additional repository section
        st.subheader("➕ Add Repository")
        st.markdown("Add more repositories to your analysis:")
        
        repo_url = st.text_input(
            "Repository URL:",
            placeholder="https://github.com/username/repository",
            help="Enter the full URL of a public Git repository",
            key="new_repo_url"  # Add unique key
        )
        
        if st.button("Add Repository", type="primary"):
            if repo_url:
                add_additional_repository(repo_url)
            else:
                st.error("Please enter a repository URL")
        
        st.markdown("---")
        
        # Management options
        st.subheader("🔧 Management")
        
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.rag_chain.clear_memory()
            st.success("Chat history cleared!")
            st.rerun()
        
        if st.button("🔄 Refresh Stats", help="Refresh repository statistics"):
            st.rerun()
        
        if st.button("⚠️ Reset Everything", help="Clear all data and restart"):
            if st.session_state.startup_processor.vector_store.clear_vector_store():
                st.session_state.app_initialized = False
                st.session_state.chat_history = []
                st.session_state.repositories_info = []
                st.success("Everything cleared! Restarting...")
                st.rerun()
            else:
                st.error("Failed to reset")

def add_additional_repository(repo_url: str):
    """Add an additional repository"""
    progress_placeholder = st.empty()
    
    def progress_callback(message):
        progress_placeholder.info(message)
    
    try:
        with st.spinner("Adding repository..."):
            result = st.session_state.startup_processor.add_additional_repository(repo_url, progress_callback)
            
            if result['success']:
                progress_placeholder.success(f"✅ Successfully added '{result['repo_name']}' with {result['documents_added']} document chunks!")
                
                # Update repositories info
                new_repo = {
                    'name': result['repo_name'],
                    'url': repo_url,
                    'documents': result['documents_added'],
                    'files': 'N/A'
                }
                st.session_state.repositories_info.append(new_repo)
                
                # Force refresh of the entire page to update stats
                st.balloons()
                time.sleep(1)  # Brief pause to show success
                st.rerun()
            else:
                progress_placeholder.error(f"Failed to add repository: {result['error']}")
                
    except Exception as e:
        progress_placeholder.error(f"Error adding repository: {str(e)}")

def main_chat_interface():
    """Main chat interface"""
    if not st.session_state.app_initialized:
        return
    
    st.header("🤖 AI Code Analyzer Chat")
    
    # Show status (refresh stats each time)
    stats = st.session_state.startup_processor.vector_store.get_collection_stats()
    total_docs = stats.get('total_documents', 0)
    repo_count = stats.get('repository_count', 0)
    
    st.success(f"✅ Ready to analyze! {total_docs} documents from {repo_count} repositories loaded.")
    
    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        
        with st.chat_message("assistant"):
            st.write(chat["answer"])
            
            if chat.get("sources"):
                with st.expander("📄 Source Files"):
                    for source in chat["sources"]:
                        st.code(f"{source['repository']}/{source['file_path']}", language="text")
                        st.caption(f"Preview: {source['chunk_preview']}")
    
    # Chat input
    if question := st.chat_input("Ask about your codebase..."):
        # Display user question
        with st.chat_message("user"):
            st.write(question)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing codebase..."):
                response = st.session_state.rag_chain.ask_question(question)
                
                if response['success']:
                    st.write(response['answer'])
                    
                    # Store in chat history
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": response['answer'],
                        "sources": response['source_documents']
                    })
                    
                    # Show sources
                    if response['source_documents']:
                        with st.expander("📄 Source Files"):
                            for source in response['source_documents']:
                                st.code(f"{source['repository']}/{source['file_path']}", language="text")
                                st.caption(f"Preview: {source['chunk_preview']}")
                else:
                    st.error(f"Error: {response.get('error', 'Unknown error')}")

def main():
    """Main application"""
    # Initialize session state
    initialize_session_state()
    
    # Handle startup initialization
    if not st.session_state.app_initialized:
        startup_initialization()
        return
    
    # Main app interface
    st.title("🔍 AI Code Analyzer")
    st.markdown("*Ask questions about your codebase using natural language*")
    
    # Sidebar
    sidebar_management()
    
    # Main chat interface
    main_chat_interface()

if __name__ == "__main__":
    main()
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from dataclasses import dataclass
import uuid
import asyncio
import os
import tempfile
import shutil
import logging
import aiofiles
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

from .pdf_processing_utils import (
    extract_document_content,
    consolidate_content_with_ai,
    html_to_pdf,
    AZURE_AVAILABLE
)

load_dotenv()

router = APIRouter(prefix="/api/pdf-consolidation", tags=["PDF Consolidation"])
log = logging.getLogger(__name__)
task_storage = {}

# Azure OpenAI configuration
try:
    client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_VERSION", "2024-02-01")
    )
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

class ConsolidationResult(BaseModel):
    task_id: str
    status: str
    message: str
    downloadUrl: Optional[str] = None

class PDFUploadRequest(BaseModel):
    userEmail: str

class TaskStatus(BaseModel):
    task_id: str
    status: str
    message: str
    downloadUrl: Optional[str] = None
    error: Optional[str] = None

@dataclass
class FileData:
    filename: str
    content: bytes
    content_type: str = "application/pdf"

@router.post("/process")
async def process_pdf_consolidation(
    files: List[UploadFile] = File(...),
    userEmail: str = Form(...)
):
    """Process multiple PDF files for consolidation."""
    task_id = str(uuid.uuid4())
    log.info(f"Starting PDF consolidation task {task_id} for user {userEmail}")
    
    # Initialize task status
    task_storage[task_id] = TaskStatus(
        task_id=task_id,
        status="processing",
        message="Starting PDF consolidation..."
    )
    
    # Validate input
    if len(files) < 2:
        task_storage[task_id] = TaskStatus(
            task_id=task_id,
            status="failed",
            message="At least 2 PDF files are required for consolidation",
            error="Insufficient files"
        )
        raise HTTPException(status_code=400, detail="At least 2 PDF files are required for consolidation")
    
    # Validate file types and read file contents
    file_data_list = []
    try:
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                task_storage[task_id] = TaskStatus(
                    task_id=task_id,
                    status="failed",
                    message=f"File {file.filename} is not a PDF",
                    error="Invalid file type"
                )
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
            
            # Read file content while request is active
            await file.seek(0)
            content = await file.read()
            
            file_data_list.append(FileData(
                filename=file.filename,
                content=content,
                content_type=file.content_type or "application/pdf"
            ))
            
    except Exception as e:
        task_storage[task_id] = TaskStatus(
            task_id=task_id,
            status="failed",
            message=f"Error reading files: {str(e)}",
            error="File read error"
        )
        raise HTTPException(status_code=500, detail=f"Error reading files: {str(e)}")
    
    # Start background processing
    asyncio.create_task(process_pdfs_background(task_id, file_data_list, userEmail))
    
    return ConsolidationResult(
        task_id=task_id,
        status="processing",
        message="PDF consolidation started. Use the task ID to check status and download results."
    )

async def process_pdfs_background(task_id: str, file_data_list: List[FileData], userEmail: str):
    """Background task to process PDFs and generate consolidated output."""
    temp_dir = None
    try:
        print(f"DEBUG: Starting background task for {task_id}")
        
        # Update status
        task_storage[task_id].status = "processing"
        task_storage[task_id].message = "Extracting content from PDF files... (0%)"
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        log.info(f"Created temporary directory: {temp_dir}")
        
        # Extract content from each PDF
        extracted_contents = []
        total_files = len(file_data_list)
        
        for i, file_data in enumerate(file_data_list):
            progress_percent = int((i / total_files) * 40)
            task_storage[task_id].message = f"Processing file {i+1} of {total_files}: {file_data.filename} ({progress_percent}%)"
            
            # Save uploaded file temporarily
            temp_file_path = os.path.join(temp_dir, file_data.filename)
            
            async with aiofiles.open(temp_file_path, 'wb') as temp_file:
                await temp_file.write(file_data.content)
            
            # Extract content
            extracted_data = await extract_document_content("application/pdf", temp_file_path)
            
            if isinstance(extracted_data, dict):
                # Clean and validate extracted data
                cleaned_data = {
                    'filename': file_data.filename,
                    'content': str(extracted_data.get('content', '')).strip(),
                    'tables': validate_tables(extracted_data.get('tables', [])),
                    'figures': validate_figures(extracted_data.get('figures', [])),
                    'hyperlinks': validate_hyperlinks(extracted_data.get('hyperlinks', [])),
                }
                extracted_contents.append(cleaned_data)
            else:
                extracted_contents.append({
                    'filename': file_data.filename,
                    'content': str(extracted_data).strip() if extracted_data else '',
                    'tables': [],
                    'figures': [],
                    'hyperlinks': [],
                })
        
        # Consolidate content using enhanced processing
        task_storage[task_id].message = "Consolidating content with enhanced formatting... (50%)"
        consolidated_content = await consolidate_content_with_ai(extracted_contents)
        
        # Generate consolidated PDF
        task_storage[task_id].message = "Generating consolidated PDF... (75%)"
        consolidated_pdf_path = os.path.join(temp_dir, f"consolidated_{task_id}.pdf")
        
        # Use HTML to PDF conversion
        html_to_pdf(consolidated_content, consolidated_pdf_path, temp_dir)
        
        # Save to static directory for download
        task_storage[task_id].message = "Preparing download... (90%)"
        static_dir = os.path.join(os.path.dirname(__file__), "..", "static", "downloads")
        os.makedirs(static_dir, exist_ok=True)
        
        final_pdf_path = os.path.join(static_dir, f"consolidated_{task_id}.pdf")
        shutil.copy2(consolidated_pdf_path, final_pdf_path)
        
        # Update task status to completed
        download_url = f"/api/pdf-consolidation/download/{task_id}"
        task_storage[task_id] = TaskStatus(
            task_id=task_id,
            status="completed",
            message="PDF consolidation completed successfully! Enhanced formatting applied. (100%)",
            downloadUrl=download_url
        )
        
        log.info(f"PDF consolidation completed successfully. Task ID: {task_id}")
        
    except Exception as e:
        error_msg = f"PDF consolidation failed: {str(e)}"
        log.error(error_msg)
        import traceback
        traceback.print_exc()
        
        task_storage[task_id] = TaskStatus(
            task_id=task_id,
            status="failed",
            message=error_msg,
            error=str(e)
        )
    
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                log.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                log.warning(f"Failed to clean up temporary directory: {str(e)}")

def validate_tables(tables):
    """Validate and clean table data."""
    if not tables:
        return []
    
    validated_tables = []
    for table in tables:
        if isinstance(table, dict) and table.get('grid'):
            # Clean table grid data
            grid = table.get('grid', [])
            cleaned_grid = []
            for row in grid:
                if isinstance(row, list):
                    cleaned_row = [str(cell).strip() if cell else '' for cell in row]
                    if any(cleaned_row):  # Only keep non-empty rows
                        cleaned_grid.append(cleaned_row)
            
            if cleaned_grid:
                validated_tables.append({
                    'grid': cleaned_grid,
                    'has_header': table.get('has_header', len(cleaned_grid) > 0),
                    'row_count': len(cleaned_grid),
                    'column_count': len(cleaned_grid[0]) if cleaned_grid else 0
                })
    
    return validated_tables

def validate_figures(figures):
    """Validate and clean figure data."""
    if not figures:
        return []
    
    validated_figures = []
    for figure in figures:
        if isinstance(figure, dict):
            caption = str(figure.get('caption', '')).strip()
            if caption or figure.get('url') or figure.get('content'):
                validated_figures.append({
                    'id': str(figure.get('id', f'fig_{len(validated_figures)+1}')),
                    'caption': caption,
                    'content': str(figure.get('content', '')).strip(),
                    'url': str(figure.get('url', '')).strip()
                })
    
    return validated_figures

def validate_hyperlinks(hyperlinks):
    """Validate and clean hyperlink data."""
    if not hyperlinks:
        return []
    
    validated_links = []
    for link in hyperlinks:
        if isinstance(link, dict):
            text = str(link.get('text', '')).strip()
            url = str(link.get('url', '')).strip()
            if text and url and url.startswith(('http://', 'https://', 'mailto:')):
                validated_links.append({
                    'text': text,
                    'url': url
                })
    
    return validated_links

@router.get("/download/{task_id}")
async def download_consolidated_pdf(task_id: str):
    """Download the consolidated PDF for a completed task."""
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = task_storage[task_id]
    
    if task.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Task is not completed. Current status: {task.status}"
        )
    
    # Find the PDF file
    static_dir = os.path.join(os.path.dirname(__file__), "..", "static", "downloads")
    pdf_path = os.path.join(static_dir, f"consolidated_{task_id}.pdf")
    
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF file not found")
    
    return FileResponse(
        path=pdf_path,
        filename=f"consolidated_document_{task_id}.pdf",
        media_type="application/pdf"
    )

@router.get("/status/{task_id}")
async def get_consolidation_status(task_id: str):
    """Get the status of a PDF consolidation task."""
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = task_storage[task_id]
    
    return {
        "task_id": task.task_id,
        "status": task.status,
        "message": task.message,
        "downloadUrl": task.downloadUrl,
        "error": task.error
    }

from typing import List, Dict, Any, Optional
import logging, html, re, os, uuid, asyncio, tempfile, shutil, aiofiles
from dataclasses import dataclass
from bs4 import BeautifulSoup
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

# Initialize components
router = APIRouter(prefix="/api/pdf-consolidation", tags=["PDF Consolidation"])
task_storage = {}

# Check Azure availability
try:
    from file_processing.document_processing import extract_document_content as azure_extract
    AZURE_AVAILABLE = True
except:
    AZURE_AVAILABLE = False
    azure_extract = None

try:
    client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_VERSION", "2024-02-01")
    )
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False

# Models
class ConsolidationResult(BaseModel):
    task_id: str
    status: str
    message: str
    downloadUrl: Optional[str] = None

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

# Core document extraction
async def extract_document_content(file_type: str, document_path: str) -> dict:
    """Extract content from PDF document using Azure or mock implementation."""
    if AZURE_AVAILABLE and azure_extract:
        return await azure_extract(file_type, document_path)
    
    # Mock implementation
    return {
        'content': f"""# Sample Document Content from {document_path}
## Introduction
This is sample content extracted from the PDF document.
## Key Points
- Point 1: Important information
- Point 2: More details
- Point 3: Additional context
## Conclusion
This concludes the sample document content.""",
        'tables': [{
            'row_count': 3, 'column_count': 3, 'has_header': True,
            'grid': [['Column 1', 'Column 2', 'Column 3'], 
                    ['Data A', 'Data B', 'Data C'], 
                    ['Data D', 'Data E', 'Data F']]
        }],
        'figures': [{'id': 'sample_figure_1', 'caption': 'Sample Chart', 'content': 'Chart data', 'url': 'placeholder://chart1'}],
        'hyperlinks': [{'text': 'Example Link', 'url': 'https://example.com'}]
    }

# HTML generation utilities
def create_html_table_with_context(table_data: dict, table_index: int = None) -> str:
    """Create properly formatted HTML table."""
    try:
        grid = table_data.get('grid', [])
        if not grid:
            return ""
        
        has_header = table_data.get('has_header', True)
        table_html = f'''<div class="table-container" style="margin: 20px 0; overflow-x: auto;">
            {f'<h4 style="color: #1B5E20; margin-bottom: 10px; font-weight: bold;">Table {table_index}</h4>' if table_index else ''}
            <table style="border-collapse: collapse; width: 100%; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">'''
        
        if has_header and len(grid) > 0:
            table_html += '<thead><tr>'
            for cell in grid[0]:
                escaped_cell = html.escape(str(cell).strip()) if cell else ''
                table_html += f'<th style="background-color: #1B5E20; color: white; padding: 12px; border: 1px solid #ddd;">{escaped_cell}</th>'
            table_html += '</tr></thead>'
            data_rows = grid[1:]
        else:
            data_rows = grid
        
        table_html += '<tbody>'
        for i, row in enumerate(data_rows):
            bg_color = '#F8F9FA' if i % 2 == 0 else 'white'
            table_html += f'<tr style="background-color: {bg_color};">'
            for cell in row:
                escaped_cell = html.escape(str(cell).strip()) if cell else ''
                table_html += f'<td style="padding: 10px 12px; border: 1px solid #ddd;">{escaped_cell}</td>'
            table_html += '</tr>'
        table_html += '</tbody></table></div>'
        return table_html
    except Exception as e:
        log.warning(f"Error creating HTML table: {e}")
        return f'<div style="color: red;">Error displaying table: {html.escape(str(e))}</div>'

def create_html_figure_with_context(figure_data: dict, figure_index: int = None) -> str:
    """Create HTML figure representation."""
    try:
        caption = figure_data.get('caption', 'Untitled Figure')
        url = figure_data.get('url', '')
        content = figure_data.get('content', '')
        
        if url and not url.startswith('placeholder://'):
            return f'''<div class="figure-container" style="text-align: center; margin: 20px 0;">
                {f'<h4 style="color: #1B5E20;">Figure {figure_index}</h4>' if figure_index else ''}
                <img src="{html.escape(url)}" alt="{html.escape(caption)}" style="max-width: 100%;">
                <p><strong>Caption:</strong> {html.escape(caption)}</p></div>'''
        else:
            return f'''<div class="figure-placeholder" style="border: 2px dashed #4CAF50; padding: 20px; text-align: center; margin: 20px 0;">
                {f'<h4 style="color: #2E7D32;">Figure {figure_index}</h4>' if figure_index else ''}
                <div style="font-size: 48px; color: #4CAF50;">[CHART]</div>
                <strong>{html.escape(caption)}</strong></div>'''
    except Exception as e:
        log.warning(f"Error creating HTML figure: {e}")
        return f'<div style="color: red;">Error displaying figure</div>'

def create_html_links_section(links: List[dict]) -> str:
    """Create links section HTML."""
    if not links:
        return ""
    
    links_html = '''<div class="links-section" style="margin: 20px 0; padding: 15px; background-color: #f8f9fa;">
        <h4 style="color: #1B5E20;">Related Links</h4><ul>'''
    
    for link in links:
        text = link.get('text', '').strip()
        url = link.get('url', '').strip()
        if text and url:
            links_html += f'<li><a href="{html.escape(url)}" target="_blank">{html.escape(text)}</a></li>'
    
    return links_html + '</ul></div>'

def convert_markdown_to_html(content: str) -> str:
    """Convert markdown to HTML."""
    if not content:
        return ""
    
    # Convert headers
    content = re.sub(r'^### (.*?)$', r'<h3 style="color: #388E3C;">\1</h3>', content, flags=re.MULTILINE)
    content = re.sub(r'^## (.*?)$', r'<h2 style="color: #2E7D32;">\1</h2>', content, flags=re.MULTILINE)
    content = re.sub(r'^# (.*?)$', r'<h1 style="color: #1B5E20;">\1</h1>', content, flags=re.MULTILINE)
    
    # Convert lists
    list_items = re.findall(r'^- (.*?)$', content, flags=re.MULTILINE)
    for item in list_items:
        content = content.replace(f'- {item}', f'<li>{item}</li>')
    content = re.sub(r'(<li>.*?</li>\s*)+', r'<ul>\g<0></ul>', content, flags=re.DOTALL)
    
    # Convert paragraphs
    paragraphs = re.split(r'\n\s*\n', content)
    html_paragraphs = []
    for para in paragraphs:
        para = para.strip()
        if para and not (para.startswith('<') or para.startswith('<ul') or para.startswith('<h')):
            html_paragraphs.append(f'<p style="margin: 10px 0; line-height: 1.6;">{para}</p>')
        elif para:
            html_paragraphs.append(para)
    
    return '\n'.join(html_paragraphs)

# Content consolidation
async def consolidate_content_with_ai(extracted_contents: List[Dict[str, Any]]) -> str:
    """Consolidate content with enhanced integration."""
    consolidated_content = '''<div style="max-width: 1200px; margin: 0 auto; font-family: 'Segoe UI', sans-serif;">
        <h1 style="color: #1B5E20; text-align: center; border-bottom: 3px solid #1B5E20;">Consolidated Document</h1>'''
    
    for doc_index, pdf_content in enumerate(extracted_contents, 1):
        file_name = pdf_content.get('filename', f'Document_{doc_index}')
        content = pdf_content.get('content', '')
        tables = pdf_content.get('tables', [])
        figures = pdf_content.get('figures', [])
        hyperlinks = pdf_content.get('hyperlinks', [])
        
        consolidated_content += f'''<div class="document-section" style="margin: 30px 0; padding: 20px; border: 1px solid #e0e0e0;">
            <h2 style="color: #1B5E20; border-bottom: 2px solid #4CAF50;">[FILE] {html.escape(file_name)}</h2>'''
        
        if content:
            processed_content = convert_markdown_to_html(content)
            consolidated_content += f'<div class="main-content">{processed_content}</div>'
        
        # Add tables
        if tables:
            consolidated_content += '<div class="tables-section"><h3 style="color: #2E7D32;">[TABLES]</h3>'
            for i, table in enumerate(tables, 1):
                consolidated_content += create_html_table_with_context(table, i)
            consolidated_content += '</div>'
        
        # Add figures
        if figures:
            consolidated_content += '<div class="figures-section"><h3 style="color: #2E7D32;">[FIGURES]</h3>'
            for i, figure in enumerate(figures, 1):
                consolidated_content += create_html_figure_with_context(figure, i)
            consolidated_content += '</div>'
        
        # Add links
        if hyperlinks:
            consolidated_content += create_html_links_section(hyperlinks)
        
        consolidated_content += '</div>'
    
    consolidated_content += '</div>'
    log.info(f"Consolidation completed with {len(extracted_contents)} documents")
    return consolidated_content

# PDF generation utilities
def safe_paragraph(text: str, style) -> Paragraph:
    """Create safe paragraph with HTML handling."""
    if not text or not text.strip():
        return Spacer(1, 6)
    
    try:
        text = text.strip()
        # Clean HTML tags
        text = re.sub(r'<div[^>]*>', '', text)
        text = re.sub(r'</div>', '', text)
        text = re.sub(r'<span[^>]*>', '', text)
        text = re.sub(r'</span>', '', text)
        text = re.sub(r'style="[^"]*"', '', text)
        
        # Convert links
        text = re.sub(r'<a\s+href="([^"]*)"[^>]*>(.*?)</a>', r'<link href="\1"><u>\2</u></link>', text)
        text = re.sub(r'<(?:strong|b)>(.*?)</(?:strong|b)>', r'<b>\1</b>', text)
        text = re.sub(r'<(?:em|i)>(.*?)</(?:em|i)>', r'<i>\1</i>', text)
        text = re.sub(r'<(?!/?[biulk]|/?font|/?link)[^>]*?>', '', text)
        
        if len(text) > 2000:
            text = text[:1997] + "..."
        
        return Paragraph(text, style)
    except Exception as e:
        log.warning(f"Failed to create paragraph: {e}")
        plain_text = re.sub(r'<[^>]+>', '', text)
        return Paragraph(html.escape(plain_text[:997] + "..." if len(plain_text) > 1000 else plain_text), style)

def create_table_from_html(table_html: str) -> Optional[Table]:
    """Create ReportLab Table from HTML."""
    try:
        soup = BeautifulSoup(table_html, 'html.parser')
        table_tag = soup.find('table')
        if not table_tag:
            return None
        
        rows = table_tag.find_all('tr')
        table_data = []
        
        for row in rows:
            cells = row.find_all(['th', 'td'])
            row_data = [cell.get_text(strip=True)[:100] for cell in cells]
            if row_data:
                table_data.append(row_data)
        
        if not table_data:
            return None
        
        max_cols = max(len(row) for row in table_data)
        for row in table_data:
            while len(row) < max_cols:
                row.append("")
        
        table = Table(table_data, colWidths=[6.5 * inch / max_cols] * max_cols)
        
        style = [
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]
        
        # Header styling
        if rows and rows[0].find('th'):
            style.extend([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1B5E20')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ])
        
        table.setStyle(TableStyle(style))
        return table
    except Exception as e:
        log.warning(f"Error creating table: {e}")
        return None

def html_to_pdf(html_content: str, output_path: str, temp_dir: str = None):
    """Convert HTML to PDF using ReportLab."""
    try:
        doc = SimpleDocTemplate(output_path, pagesize=A4, topMargin=72, bottomMargin=72, leftMargin=72, rightMargin=72)
        styles = getSampleStyleSheet()
        story = []
        
        # Custom styles
        title_style = ParagraphStyle('CustomTitle', parent=styles['Title'], fontSize=18, spaceAfter=18, 
                                   alignment=TA_CENTER, textColor=colors.HexColor('#1B5E20'))
        heading1_style = ParagraphStyle('CustomHeading1', parent=styles['Heading1'], fontSize=14, spaceAfter=12, 
                                      textColor=colors.HexColor('#1B5E20'))
        heading2_style = ParagraphStyle('CustomHeading2', parent=styles['Heading2'], fontSize=12, spaceAfter=10, 
                                      textColor=colors.HexColor('#2E7D32'))
        normal_style = ParagraphStyle('CustomNormal', parent=styles['Normal'], fontSize=10, spaceAfter=8, 
                                    alignment=TA_JUSTIFY, textColor=colors.HexColor('#212121'))
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        for element in soup.find_all(['h1', 'h2', 'h3', 'p', 'table', 'div']):
            try:
                if element.name == 'h1':
                    story.append(safe_paragraph(element.get_text(strip=True), title_style))
                elif element.name == 'h2':
                    story.append(safe_paragraph(element.get_text(strip=True), heading1_style))
                elif element.name == 'h3':
                    story.append(safe_paragraph(element.get_text(strip=True), heading2_style))
                elif element.name == 'p':
                    story.append(safe_paragraph(str(element), normal_style))
                elif element.name == 'table':
                    table = create_table_from_html(str(element))
                    if table:
                        story.append(Spacer(1, 8))
                        story.append(table)
                        story.append(Spacer(1, 8))
            except Exception as e:
                log.warning(f"Error processing element {element.name}: {e}")
        
        if not story:
            story.append(safe_paragraph("Document processed successfully, but no readable content found.", normal_style))
        
        doc.build(story)
        log.info(f"PDF generated successfully at {output_path}")
        
    except Exception as e:
        log.error(f"Error converting HTML to PDF: {e}")
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")

# Data validation utilities
def validate_tables(tables):
    """Validate table data."""
    if not tables:
        return []
    
    validated = []
    for table in tables:
        if isinstance(table, dict) and table.get('grid'):
            grid = table.get('grid', [])
            cleaned_grid = []
            for row in grid:
                if isinstance(row, list):
                    cleaned_row = [str(cell).strip() if cell else '' for cell in row]
                    if any(cleaned_row):
                        cleaned_grid.append(cleaned_row)
            
            if cleaned_grid:
                validated.append({
                    'grid': cleaned_grid,
                    'has_header': table.get('has_header', True),
                    'row_count': len(cleaned_grid),
                    'column_count': len(cleaned_grid[0]) if cleaned_grid else 0
                })
    return validated

def validate_figures(figures):
    """Validate figure data."""
    if not figures:
        return []
    
    validated = []
    for figure in figures:
        if isinstance(figure, dict):
            caption = str(figure.get('caption', '')).strip()
            if caption or figure.get('url') or figure.get('content'):
                validated.append({
                    'id': str(figure.get('id', f'fig_{len(validated)+1}')),
                    'caption': caption,
                    'content': str(figure.get('content', '')).strip(),
                    'url': str(figure.get('url', '')).strip()
                })
    return validated

def validate_hyperlinks(hyperlinks):
    """Validate hyperlink data."""
    if not hyperlinks:
        return []
    
    validated = []
    for link in hyperlinks:
        if isinstance(link, dict):
            text = str(link.get('text', '')).strip()
            url = str(link.get('url', '')).strip()
            if text and url and url.startswith(('http://', 'https://', 'mailto:')):
                validated.append({'text': text, 'url': url})
    return validated

# Background processing
async def process_pdfs_background(task_id: str, file_data_list: List[FileData], userEmail: str):
    """Background task to process PDFs."""
    temp_dir = None
    try:
        task_storage[task_id].message = "Extracting content from PDF files..."
        temp_dir = tempfile.mkdtemp()
        
        extracted_contents = []
        for i, file_data in enumerate(file_data_list):
            progress = int((i / len(file_data_list)) * 40)
            task_storage[task_id].message = f"Processing {file_data.filename} ({progress}%)"
            
            temp_file_path = os.path.join(temp_dir, file_data.filename)
            async with aiofiles.open(temp_file_path, 'wb') as f:
                await f.write(file_data.content)
            
            extracted_data = await extract_document_content("application/pdf", temp_file_path)
            
            if isinstance(extracted_data, dict):
                cleaned_data = {
                    'filename': file_data.filename,
                    'content': str(extracted_data.get('content', '')).strip(),
                    'tables': validate_tables(extracted_data.get('tables', [])),
                    'figures': validate_figures(extracted_data.get('figures', [])),
                    'hyperlinks': validate_hyperlinks(extracted_data.get('hyperlinks', [])),
                }
            else:
                cleaned_data = {
                    'filename': file_data.filename,
                    'content': str(extracted_data).strip() if extracted_data else '',
                    'tables': [], 'figures': [], 'hyperlinks': []
                }
            extracted_contents.append(cleaned_data)
        
        task_storage[task_id].message = "Consolidating content... (50%)"
        consolidated_content = await consolidate_content_with_ai(extracted_contents)
        
        task_storage[task_id].message = "Generating PDF... (75%)"
        consolidated_pdf_path = os.path.join(temp_dir, f"consolidated_{task_id}.pdf")
        html_to_pdf(consolidated_content, consolidated_pdf_path, temp_dir)
        
        # Save for download
        static_dir = os.path.join(os.path.dirname(__file__), "..", "static", "downloads")
        os.makedirs(static_dir, exist_ok=True)
        final_pdf_path = os.path.join(static_dir, f"consolidated_{task_id}.pdf")
        shutil.copy2(consolidated_pdf_path, final_pdf_path)
        
        task_storage[task_id] = TaskStatus(
            task_id=task_id, status="completed",
            message="PDF consolidation completed successfully! (100%)",
            downloadUrl=f"/api/pdf-consolidation/download/{task_id}"
        )
        
    except Exception as e:
        task_storage[task_id] = TaskStatus(
            task_id=task_id, status="failed",
            message=f"PDF consolidation failed: {str(e)}", error=str(e)
        )
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                log.warning(f"Failed to clean up: {e}")

# API Endpoints
@router.post("/process")
async def process_pdf_consolidation(files: List[UploadFile] = File(...), userEmail: str = Form(...)):
    """Process multiple PDF files for consolidation."""
    task_id = str(uuid.uuid4())
    
    task_storage[task_id] = TaskStatus(task_id=task_id, status="processing", 
                                     message="Starting PDF consolidation...")
    
    if len(files) < 2:
        task_storage[task_id] = TaskStatus(task_id=task_id, status="failed", 
                                         message="At least 2 PDF files required", error="Insufficient files")
        raise HTTPException(status_code=400, detail="At least 2 PDF files required")
    
    file_data_list = []
    try:
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                task_storage[task_id] = TaskStatus(task_id=task_id, status="failed", 
                                                 message=f"File {file.filename} is not a PDF", error="Invalid file type")
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
            
            await file.seek(0)
            content = await file.read()
            file_data_list.append(FileData(filename=file.filename, content=content, 
                                         content_type=file.content_type or "application/pdf"))
            
    except Exception as e:
        task_storage[task_id] = TaskStatus(task_id=task_id, status="failed", 
                                         message=f"Error reading files: {e}", error="File read error")
        raise HTTPException(status_code=500, detail=f"Error reading files: {e}")
    
    asyncio.create_task(process_pdfs_background(task_id, file_data_list, userEmail))
    
    return ConsolidationResult(task_id=task_id, status="processing", 
                             message="PDF consolidation started. Use task ID to check status.")

@router.get("/download/{task_id}")
async def download_consolidated_pdf(task_id: str):
    """Download consolidated PDF."""
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = task_storage[task_id]
    if task.status != "completed":
        raise HTTPException(status_code=400, detail=f"Task not completed. Status: {task.status}")
    
    static_dir = os.path.join(os.path.dirname(__file__), "..", "static", "downloads")
    pdf_path = os.path.join(static_dir, f"consolidated_{task_id}.pdf")
    
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF file not found")
    
    return FileResponse(path=pdf_path, filename=f"consolidated_document_{task_id}.pdf", 
                       media_type="application/pdf")

@router.get("/status/{task_id}")
async def get_consolidation_status(task_id: str):
    """Get task status."""
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = task_storage[task_id]
    return {"task_id": task.task_id, "status": task.status, "message": task.message, 
            "downloadUrl": task.downloadUrl, "error": task.error}

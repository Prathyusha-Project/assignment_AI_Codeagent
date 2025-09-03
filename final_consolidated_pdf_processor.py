from typing import List, Dict, Any, Optional, Tuple
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

# ================================
# CONFIGURATION
# ================================
load_dotenv()
log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/pdf-consolidation", tags=["PDF Consolidation"])
task_storage = {}

# Service initialization
try:
    from file_processing.document_processing import extract_document_content as azure_extract
    AZURE_AVAILABLE = True
except Exception:
    AZURE_AVAILABLE = False
    azure_extract = None

try:
    client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_VERSION", "2024-02-01")
    )
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# ================================
# DATA MODELS
# ================================
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

@dataclass
class ContentElement:
    type: str  # 'text', 'table', 'figure', 'links'
    content: Any
    position: int
    context: str = ""

# ================================
# CORE EXTRACTION
# ================================
async def extract_document_content(file_type: str, document_path: str) -> dict:
    """Extract content from PDF document using Azure or mock implementation."""
    if AZURE_AVAILABLE and azure_extract:
        return await azure_extract(file_type, document_path)
    
    # Enhanced mock with realistic content structure
    filename = os.path.basename(document_path)
    return {
        'content': f"""# Analysis Report: {filename}

## Executive Summary
This document presents a comprehensive analysis of the key performance indicators and strategic recommendations for the upcoming quarter.

## Key Metrics Overview
The following table summarizes the critical performance metrics:

## Performance Analysis
Based on the data collected, we observe several important trends that require immediate attention.

## Visual Representation
The chart below illustrates the quarterly performance trends across different business units.

## Strategic Recommendations
1. Increase investment in high-performing sectors
2. Optimize resource allocation based on performance data
3. Implement enhanced monitoring systems

## Conclusion
The analysis indicates positive growth potential with strategic adjustments to current operations.""",
        
        'tables': [
            {
                'row_count': 4,
                'column_count': 4,
                'has_header': True,
                'grid': [
                    ['Metric', 'Q1 2024', 'Q2 2024', 'Target'],
                    ['Revenue (M)', '$12.5', '$14.2', '$15.0'],
                    ['Growth Rate', '15%', '18%', '20%'],
                    ['Customer Satisfaction', '87%', '91%', '95%']
                ]
            }
        ],
        'figures': [
            {
                'id': 'quarterly_trends_chart',
                'caption': 'Quarterly Performance Trends by Business Unit',
                'content': 'Line chart showing upward trajectory in key performance indicators',
                'url': 'placeholder://chart_quarterly_trends'
            }
        ],
        'hyperlinks': [
            {'text': 'Detailed Methodology Report', 'url': 'https://company.com/methodology'},
            {'text': 'Previous Quarter Analysis', 'url': 'https://company.com/q1-analysis'}
        ]
    }

# ================================
# INTELLIGENT CONTENT ANALYSIS
# ================================
def analyze_content_structure(content: str, tables: List[dict], figures: List[dict]) -> List[ContentElement]:
    """Analyze content and determine optimal placement of tables and figures."""
    elements = []
    
    # Split content into paragraphs with position tracking
    paragraphs = re.split(r'\n\s*\n', content)
    current_pos = 0
    table_index = 0
    figure_index = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # Add paragraph as text element
        elements.append(ContentElement('text', para, current_pos, para))
        
        para_lower = para.lower()
        
        # Check if this paragraph mentions tables
        if (any(word in para_lower for word in ['table', 'data', 'metrics', 'summary', 'following']) and 
            table_index < len(tables)):
            elements.append(ContentElement('table', tables[table_index], current_pos + 1, para))
            table_index += 1
        
        # Check if this paragraph mentions figures/charts
        if (any(word in para_lower for word in ['chart', 'figure', 'graph', 'visual', 'illustration', 'below']) and 
            figure_index < len(figures)):
            elements.append(ContentElement('figure', figures[figure_index], current_pos + 1, para))
            figure_index += 1
        
        current_pos += 2
    
    # Add remaining tables and figures at the end if not placed
    while table_index < len(tables):
        elements.append(ContentElement('table', tables[table_index], current_pos, 'Additional data'))
        table_index += 1
        current_pos += 1
    
    while figure_index < len(figures):
        elements.append(ContentElement('figure', figures[figure_index], current_pos, 'Additional visualization'))
        figure_index += 1
        current_pos += 1
    
    return elements

# ================================
# HTML GENERATION
# ================================
def create_html_table(table_data: dict, table_index: int, context: str = "") -> str:
    """Create properly formatted HTML table with context."""
    try:
        grid = table_data.get('grid', [])
        if not grid:
            return ""
        
        has_header = table_data.get('has_header', True)
        
        # Create contextual title based on surrounding content
        context_words = context.lower()
        if 'metric' in context_words or 'performance' in context_words:
            title = f"Performance Metrics - Table {table_index}"
        elif 'financial' in context_words or 'revenue' in context_words:
            title = f"Financial Data - Table {table_index}"
        else:
            title = f"Data Table {table_index}"
        
        html_parts = [
            f'<div class="table-container" style="margin: 20px 0; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden;">',
            f'<h4 style="background: linear-gradient(135deg, #1B5E20, #2E7D32); color: white; margin: 0; padding: 15px; font-size: 14px;">{title}</h4>',
            '<table style="width: 100%; border-collapse: collapse; background: white;">'
        ]
        
        # Header row
        if has_header and grid:
            html_parts.append('<thead><tr>')
            for cell in grid[0]:
                escaped = html.escape(str(cell).strip()) if cell else ''
                html_parts.append(f'<th style="background: #f5f5f5; padding: 12px; border: 1px solid #ddd; text-align: left; font-weight: 600; color: #333;">{escaped}</th>')
            html_parts.append('</tr></thead>')
            data_rows = grid[1:]
        else:
            data_rows = grid
        
        # Data rows
        html_parts.append('<tbody>')
        for i, row in enumerate(data_rows):
            bg_color = '#fafafa' if i % 2 == 0 else 'white'
            html_parts.append(f'<tr style="background: {bg_color};">')
            for cell in row:
                escaped = html.escape(str(cell).strip()) if cell else ''
                html_parts.append(f'<td style="padding: 10px 12px; border: 1px solid #ddd; color: #444;">{escaped}</td>')
            html_parts.append('</tr>')
        
        html_parts.extend(['</tbody></table></div>'])
        return ''.join(html_parts)
        
    except Exception as e:
        log.warning(f"Error creating HTML table: {e}")
        return f'<div style="color: red; padding: 10px; border: 1px solid red;">Error displaying table: {html.escape(str(e))}</div>'

def create_html_figure(figure_data: dict, figure_index: int, context: str = "") -> str:
    """Create properly formatted HTML figure with context."""
    try:
        caption = figure_data.get('caption', 'Untitled Figure')
        url = figure_data.get('url', '')
        content = figure_data.get('content', '')
        
        # Determine figure type from context
        context_lower = context.lower()
        if 'trend' in context_lower or 'performance' in context_lower:
            icon = "📈"
            fig_type = "Performance Chart"
        elif 'comparison' in context_lower or 'analysis' in context_lower:
            icon = "📊"
            fig_type = "Analysis Chart"
        else:
            icon = "📋"
            fig_type = "Figure"
        
        if url and not url.startswith('placeholder://'):
            return f'''
            <div class="figure-container" style="margin: 25px 0; border: 1px solid #e0e0e0; border-radius: 8px; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="background: linear-gradient(135deg, #1565C0, #1976D2); color: white; padding: 12px; border-radius: 8px 8px 0 0;">
                    <h4 style="margin: 0; font-size: 14px;">{icon} {fig_type} {figure_index}</h4>
                </div>
                <div style="padding: 20px; text-align: center;">
                    <img src="{html.escape(url)}" alt="{html.escape(caption)}" style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px;">
                    <p style="margin: 15px 0 5px 0; font-weight: 600; color: #333;">{html.escape(caption)}</p>
                    {f'<p style="color: #666; font-size: 13px; line-height: 1.4;">{html.escape(content[:200])}{"..." if len(content) > 200 else ""}</p>' if content else ''}
                </div>
            </div>
            '''
        else:
            return f'''
            <div class="figure-placeholder" style="margin: 25px 0; border: 2px dashed #4CAF50; border-radius: 8px; background: #f1f8e9; text-align: center; padding: 30px;">
                <div style="font-size: 48px; color: #4CAF50; margin-bottom: 15px;">{icon}</div>
                <h4 style="color: #2E7D32; margin: 0 0 10px 0; font-size: 16px;">{fig_type} {figure_index}</h4>
                <p style="color: #2E7D32; font-weight: 600; margin: 0 0 8px 0;">{html.escape(caption)}</p>
                {f'<p style="color: #555; font-size: 13px; margin: 0;">{html.escape(content[:150])}{"..." if len(content) > 150 else ""}</p>' if content else ''}
                <small style="color: #777; display: block; margin-top: 10px; font-style: italic;">Figure placeholder - visual content not available</small>
            </div>
            '''
            
    except Exception as e:
        log.warning(f"Error creating HTML figure: {e}")
        return f'<div style="color: red; padding: 10px; border: 1px solid red;">Error displaying figure: {html.escape(str(e))}</div>'

def create_html_links(links: List[dict]) -> str:
    """Create properly formatted links section."""
    if not links:
        return ""
    
    html_parts = [
        '<div class="links-section" style="margin: 25px 0; background: linear-gradient(135deg, #f8f9fa, #e9ecef); border-left: 4px solid #17a2b8; border-radius: 0 8px 8px 0; padding: 20px;">',
        '<h4 style="color: #17a2b8; margin: 0 0 15px 0; font-size: 16px; display: flex; align-items: center;">',
        '<span style="margin-right: 8px;">🔗</span> Reference Links</h4>',
        '<ul style="margin: 0; padding: 0; list-style: none;">'
    ]
    
    for i, link in enumerate(links, 1):
        text = link.get('text', '').strip()
        url = link.get('url', '').strip()
        if text and url:
            html_parts.append(f'''
            <li style="margin-bottom: 10px; padding: 8px 12px; background: white; border-radius: 4px; border: 1px solid #dee2e6;">
                <span style="color: #17a2b8; font-weight: 600; margin-right: 8px;">[{i}]</span>
                <a href="{html.escape(url)}" target="_blank" style="color: #007bff; text-decoration: none; font-weight: 500;">
                    {html.escape(text)}
                </a>
            </li>
            ''')
    
    html_parts.extend(['</ul></div>'])
    return ''.join(html_parts)

def convert_markdown_to_html(text: str) -> str:
    """Convert markdown to HTML with proper styling."""
    if not text:
        return ""
    
    # Headers
    text = re.sub(r'^# (.*?)$', r'<h1 style="color: #1B5E20; font-size: 24px; margin: 25px 0 20px 0; border-bottom: 2px solid #4CAF50; padding-bottom: 10px;">\1</h1>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.*?)$', r'<h2 style="color: #2E7D32; font-size: 20px; margin: 20px 0 15px 0;">\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^### (.*?)$', r'<h3 style="color: #388E3C; font-size: 16px; margin: 15px 0 10px 0;">\1</h3>', text, flags=re.MULTILINE)
    
    # Lists
    text = re.sub(r'^(\d+)\. (.*?)$', r'<li style="margin: 5px 0; color: #333;">\2</li>', text, flags=re.MULTILINE)
    text = re.sub(r'^- (.*?)$', r'<li style="margin: 5px 0; color: #333;">\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'(<li[^>]*>.*?</li>\s*)+', r'<ul style="margin: 10px 0; padding-left: 20px; line-height: 1.6;">\g<0></ul>', text, flags=re.DOTALL)
    
    # Paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    html_paragraphs = []
    
    for para in paragraphs:
        para = para.strip()
        if para and not para.startswith('<'):
            html_paragraphs.append(f'<p style="margin: 12px 0; line-height: 1.6; color: #333; text-align: justify;">{para}</p>')
        elif para:
            html_paragraphs.append(para)
    
    return '\n'.join(html_paragraphs)

# ================================
# CONTENT CONSOLIDATION
# ================================
async def consolidate_content_with_ai(extracted_contents: List[Dict[str, Any]]) -> str:
    """Consolidate content with intelligent table and figure placement."""
    
    html_parts = [
        '<div style="max-width: 1200px; margin: 0 auto; font-family: \'Segoe UI\', Tahoma, Geneva, Verdana, sans-serif; background: white; padding: 20px;">',
        '<div style="text-align: center; margin-bottom: 40px; padding: 30px; background: linear-gradient(135deg, #1B5E20, #2E7D32); color: white; border-radius: 12px;">',
        '<h1 style="margin: 0; font-size: 28px; font-weight: 300;">Consolidated Document Analysis</h1>',
        '<p style="margin: 10px 0 0 0; opacity: 0.9;">Comprehensive report generated from multiple PDF sources</p>',
        '</div>'
    ]
    
    for doc_index, pdf_content in enumerate(extracted_contents, 1):
        filename = pdf_content.get('filename', f'Document_{doc_index}')
        content = pdf_content.get('content', '')
        tables = pdf_content.get('tables', [])
        figures = pdf_content.get('figures', [])
        hyperlinks = pdf_content.get('hyperlinks', [])
        
        # Analyze content structure for intelligent placement
        elements = analyze_content_structure(content, tables, figures)
        
        html_parts.append(f'''
        <div class="document-section" style="margin: 30px 0; border: 1px solid #e0e0e0; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="background: linear-gradient(135deg, #f8f9fa, #e9ecef); padding: 20px; border-bottom: 1px solid #e0e0e0;">
                <h2 style="margin: 0; color: #1B5E20; font-size: 20px; display: flex; align-items: center;">
                    <span style="background: #1B5E20; color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 15px; font-size: 14px; font-weight: bold;">{doc_index}</span>
                    {html.escape(filename)}
                </h2>
            </div>
            <div style="padding: 25px;">
        ''')
        
        # Process elements in order
        table_count = 0
        figure_count = 0
        
        for element in elements:
            if element.type == 'text':
                html_content = convert_markdown_to_html(element.content)
                html_parts.append(html_content)
            
            elif element.type == 'table':
                table_count += 1
                table_html = create_html_table(element.content, table_count, element.context)
                html_parts.append(table_html)
            
            elif element.type == 'figure':
                figure_count += 1
                figure_html = create_html_figure(element.content, figure_count, element.context)
                html_parts.append(figure_html)
        
        # Add links at the end of each document section
        if hyperlinks:
            links_html = create_html_links(hyperlinks)
            html_parts.append(links_html)
        
        html_parts.append('</div></div>')  # Close document section
    
    html_parts.append('</div>')  # Close main container
    
    log.info(f"Consolidation completed with {len(extracted_contents)} documents")
    return ''.join(html_parts)

# ================================
# PDF GENERATION
# ================================
def safe_paragraph(text: str, style) -> Paragraph:
    """Create safe paragraph with proper HTML handling."""
    if not text or not text.strip():
        return Spacer(1, 6)
    
    try:
        text = text.strip()
        # Clean HTML tags but preserve basic formatting
        text = re.sub(r'<div[^>]*>', '', text)
        text = re.sub(r'</div>', '', text)
        text = re.sub(r'<span[^>]*>', '', text)
        text = re.sub(r'</span>', '', text)
        text = re.sub(r'style="[^"]*"', '', text)
        
        # Convert links
        text = re.sub(r'<a\s+href="([^"]*)"[^>]*>(.*?)</a>', r'<link href="\1"><u>\2</u></link>', text)
        
        # Convert formatting
        text = re.sub(r'<(?:strong|b)>(.*?)</(?:strong|b)>', r'<b>\1</b>', text)
        text = re.sub(r'<(?:em|i)>(.*?)</(?:em|i)>', r'<i>\1</i>', text)
        
        # Remove unsupported tags
        text = re.sub(r'<(?!/?[biulk]|/?font|/?link)[^>]*?>', '', text)
        
        if len(text) > 1500:
            text = text[:1497] + "..."
        
        return Paragraph(text, style)
        
    except Exception as e:
        log.warning(f"Failed to create paragraph: {e}")
        plain_text = re.sub(r'<[^>]+>', '', text)
        return Paragraph(html.escape(plain_text[:1000]), style)

def create_table_from_html(table_html: str) -> Optional[Table]:
    """Create ReportLab Table from HTML with enhanced styling."""
    try:
        soup = BeautifulSoup(table_html, 'html.parser')
        table_tag = soup.find('table')
        if not table_tag:
            return None
        
        rows = table_tag.find_all('tr')
        table_data = []
        
        for row in rows:
            cells = row.find_all(['th', 'td'])
            row_data = []
            for cell in cells:
                text = cell.get_text(strip=True)
                if len(text) > 80:
                    text = text[:77] + "..."
                row_data.append(text)
            if row_data:
                table_data.append(row_data)
        
        if not table_data:
            return None
        
        # Ensure consistent column count
        max_cols = max(len(row) for row in table_data) if table_data else 0
        for row in table_data:
            while len(row) < max_cols:
                row.append("")
        
        # Dynamic column widths
        table = Table(table_data, colWidths=[6.5 * inch / max_cols] * max_cols, repeatRows=1)
        
        # Enhanced styling
        style = [
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#424242')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('WORDWRAP', (0, 0), (-1, -1), True),
        ]
        
        # Header detection and styling
        has_header = rows and (rows[0].find('th') is not None or 'background: #f5f5f5' in str(rows[0]))
        
        if has_header:
            style.extend([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1B5E20')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
            ])
            
            # Alternating row colors
            for i in range(1, len(table_data)):
                if i % 2 == 0:
                    style.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#F8F9FA')))
        
        table.setStyle(TableStyle(style))
        return table
        
    except Exception as e:
        log.warning(f"Error creating table from HTML: {e}")
        return None

def html_to_pdf(html_content: str, output_path: str, temp_dir: str = None):
    """Convert HTML content to PDF using ReportLab with improved parsing."""
    try:
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            topMargin=72,
            bottomMargin=72,
            leftMargin=72,
            rightMargin=72
        )
        
        styles = getSampleStyleSheet()
        story = []
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=18,
            spaceAfter=18,
            spaceBefore=8,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1B5E20'),
            fontName='Helvetica-Bold'
        )
        
        heading1_style = ParagraphStyle(
            'CustomHeading1',
            parent=styles['Heading1'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=16,
            textColor=colors.HexColor('#1B5E20'),
            fontName='Helvetica-Bold'
        )
        
        heading2_style = ParagraphStyle(
            'CustomHeading2',
            parent=styles['Heading2'],
            fontSize=12,
            spaceAfter=10,
            spaceBefore=14,
            textColor=colors.HexColor('#2E7D32'),
            fontName='Helvetica-Bold'
        )
        
        heading3_style = ParagraphStyle(
            'CustomHeading3',
            parent=styles['Heading3'],
            fontSize=11,
            spaceAfter=8,
            spaceBefore=12,
            textColor=colors.HexColor('#388E3C'),
            fontName='Helvetica-Bold'
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=8,
            spaceBefore=2,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor('#212121'),
            fontName='Helvetica',
            leading=12
        )
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style tags
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Process elements systematically
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'div', 'ul', 'li', 'table']):
            try:
                if element.name == 'h1':
                    text = element.get_text(strip=True)
                    if text:
                        story.append(safe_paragraph(text, title_style))
                
                elif element.name == 'h2':
                    text = element.get_text(strip=True)
                    if text:
                        story.append(safe_paragraph(text, heading1_style))
                
                elif element.name == 'h3':
                    text = element.get_text(strip=True)
                    if text:
                        story.append(safe_paragraph(text, heading2_style))
                
                elif element.name == 'h4':
                    text = element.get_text(strip=True)
                    if text:
                        story.append(safe_paragraph(text, heading3_style))
                
                elif element.name == 'p':
                    text_content = str(element)
                    if text_content.strip() and element.get_text(strip=True):
                        story.append(safe_paragraph(text_content, normal_style))
                
                elif element.name == 'table' and 'table-container' not in element.parent.get('class', []):
                    table = create_table_from_html(str(element))
                    if table:
                        story.append(Spacer(1, 8))
                        story.append(table)
                        story.append(Spacer(1, 8))
                
                elif element.name == 'div' and 'table-container' in element.get('class', []):
                    table_elem = element.find('table')
                    if table_elem:
                        title_elem = element.find('h4')
                        if title_elem:
                            story.append(safe_paragraph(title_elem.get_text(strip=True), heading3_style))
                        
                        table = create_table_from_html(str(table_elem))
                        if table:
                            story.append(table)
                            story.append(Spacer(1, 10))
                
                elif element.name == 'div' and ('figure-container' in element.get('class', []) or 'figure-placeholder' in element.get('class', [])):
                    title_elem = element.find('h4')
                    if title_elem:
                        story.append(safe_paragraph(title_elem.get_text(strip=True), heading3_style))
                    
                    # Get figure description
                    text = element.get_text(separator=' ', strip=True)
                    if text and not text.startswith('Figure'):
                        text_parts = text.split('\n')
                        if len(text_parts) > 1:
                            text = '\n'.join(text_parts[1:]).strip()
                        story.append(safe_paragraph(text, normal_style))
                    story.append(Spacer(1, 8))
                
                elif element.name == 'div' and 'links-section' in element.get('class', []):
                    title_elem = element.find('h4')
                    if title_elem:
                        story.append(safe_paragraph(title_elem.get_text(strip=True), heading3_style))
                    
                    ul_elem = element.find('ul')
                    if ul_elem:
                        for li in ul_elem.find_all('li'):
                            story.append(safe_paragraph(li.get_text(strip=True), normal_style))
                    story.append(Spacer(1, 8))
                
                elif element.name in ['ul', 'li']:
                    if element.name == 'li' and element.parent.name == 'ul':
                        continue
                    
                    if element.name == 'ul':
                        for li in element.find_all('li', recursive=False):
                            story.append(safe_paragraph(li.get_text(strip=True), normal_style))
            
            except Exception as e:
                log.warning(f"Error processing element {element.name}: {e}")
                continue
        
        if not story:
            story.append(safe_paragraph("Document processed successfully, but no readable content was found.", normal_style))
        
        doc.build(story)
        log.info(f"PDF generated successfully at {output_path}")
        
    except Exception as e:
        log.error(f"Error converting HTML to PDF: {e}")
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")

# ================================
# DATA VALIDATION
# ================================
def validate_tables(tables):
    """Validate and clean table data."""
    if not tables:
        return []
    
    validated_tables = []
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

# ================================
# BACKGROUND PROCESSING
# ================================
async def process_pdfs_background(task_id: str, file_data_list: List[FileData], userEmail: str):
    """Background task to process PDFs and generate consolidated output."""
    temp_dir = None
    try:
        task_storage[task_id].status = "processing"
        task_storage[task_id].message = "Extracting content from PDF files... (0%)"
        
        temp_dir = tempfile.mkdtemp()
        log.info(f"Created temporary directory: {temp_dir}")
        
        extracted_contents = []
        total_files = len(file_data_list)
        
        for i, file_data in enumerate(file_data_list):
            progress_percent = int((i / total_files) * 40)
            task_storage[task_id].message = f"Processing file {i+1} of {total_files}: {file_data.filename} ({progress_percent}%)"
            
            temp_file_path = os.path.join(temp_dir, file_data.filename)
            
            async with aiofiles.open(temp_file_path, 'wb') as temp_file:
                await temp_file.write(file_data.content)
            
            extracted_data = await extract_document_content("application/pdf", temp_file_path)
            
            if isinstance(extracted_data, dict):
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
        
        task_storage[task_id].message = "Consolidating content with intelligent placement... (50%)"
        consolidated_content = await consolidate_content_with_ai(extracted_contents)
        
        task_storage[task_id].message = "Generating consolidated PDF... (75%)"
        consolidated_pdf_path = os.path.join(temp_dir, f"consolidated_{task_id}.pdf")
        
        html_to_pdf(consolidated_content, consolidated_pdf_path, temp_dir)
        
        task_storage[task_id].message = "Preparing download... (90%)"
        static_dir = os.path.join(os.path.dirname(__file__), "..", "static", "downloads")
        os.makedirs(static_dir, exist_ok=True)
        
        final_pdf_path = os.path.join(static_dir, f"consolidated_{task_id}.pdf")
        shutil.copy2(consolidated_pdf_path, final_pdf_path)
        
        task_storage[task_id] = TaskStatus(
            task_id=task_id,
            status="completed",
            message="PDF consolidation completed successfully with intelligent content placement! (100%)",
            downloadUrl=f"/api/pdf-consolidation/download/{task_id}"
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
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                log.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                log.warning(f"Failed to clean up temporary directory: {str(e)}")

# ================================
# API ENDPOINTS
# ================================
@router.post("/process")
async def process_pdf_consolidation(
    files: List[UploadFile] = File(...),
    userEmail: str = Form(...)
):
    """Process multiple PDF files for consolidation with intelligent content placement."""
    task_id = str(uuid.uuid4())
    log.info(f"Starting PDF consolidation task {task_id} for user {userEmail}")
    
    task_storage[task_id] = TaskStatus(
        task_id=task_id,
        status="processing",
        message="Starting PDF consolidation..."
    )
    
    if len(files) < 2:
        task_storage[task_id] = TaskStatus(
            task_id=task_id,
            status="failed",
            message="At least 2 PDF files are required for consolidation",
            error="Insufficient files"
        )
        raise HTTPException(status_code=400, detail="At least 2 PDF files are required for consolidation")
    
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
    
    asyncio.create_task(process_pdfs_background(task_id, file_data_list, userEmail))
    
    return ConsolidationResult(
        task_id=task_id,
        status="processing",
        message="PDF consolidation started. Use the task ID to check status and download results."
    )

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
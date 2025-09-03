from typing import List, Dict, Any, Optional
import logging
import html
import re
from bs4 import BeautifulSoup
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from fastapi import HTTPException
import os

log = logging.getLogger(__name__)

# Only import document processing if Azure credentials are available
try:
    from file_processing.document_processing import extract_document_content as azure_extract
    AZURE_AVAILABLE = True
except Exception:
    AZURE_AVAILABLE = False
    azure_extract = None

async def extract_document_content(file_type: str, document_path: str) -> dict:
    """Extract content from PDF document using Azure or mock implementation."""
    if AZURE_AVAILABLE and azure_extract:
        return await azure_extract(file_type, document_path)
    else:
        # Mock implementation for testing without Azure
        sample_content = f"""
# Sample Document Content from {document_path}

## Introduction
This is sample content extracted from the PDF document.

## Key Points
- Point 1: Important information
- Point 2: More details
- Point 3: Additional context

## Data Table
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data A   | Data B   | Data C   |
| Data D   | Data E   | Data F   |

## Conclusion
This concludes the sample document content.
"""
        return {
            'content': sample_content,
            'tables': [
                {
                    'row_count': 3,
                    'column_count': 3,
                    'has_header': True,
                    'grid': [
                        ['Column 1', 'Column 2', 'Column 3'],
                        ['Data A', 'Data B', 'Data C'],
                        ['Data D', 'Data E', 'Data F']
                    ]
                }
            ],
            'figures': [
                {
                    'id': 'sample_figure_1',
                    'caption': 'Sample Chart',
                    'content': 'Chart data would be here',
                    'url': 'placeholder://chart1'
                }
            ],
            'hyperlinks': [
                {
                    'text': 'Example Link',
                    'url': 'https://example.com'
                }
            ]
        }

def create_html_table_with_context(table_data: dict, table_index: int = None) -> str:
    """Create properly formatted HTML table with better styling and context."""
    try:
        grid = table_data.get('grid', [])
        if not grid:
            return ""
        
        has_header = table_data.get('has_header', True)
        
        # Add table wrapper with better styling
        table_html = f'''
        <div class="table-container" style="margin: 20px 0; overflow-x: auto;">
            {f'<h4 style="color: #1B5E20; margin-bottom: 10px; font-weight: bold;">Table {table_index}</h4>' if table_index else ''}
            <table style="border-collapse: collapse; width: 100%; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        '''
        
        # Create header if present
        if has_header and len(grid) > 0:
            table_html += '<thead>\n<tr>'
            for cell in grid[0]:
                escaped_cell = html.escape(str(cell).strip()) if cell else ''
                table_html += f'<th style="background-color: #1B5E20; color: white; padding: 12px; border: 1px solid #ddd; text-align: left; font-weight: bold;">{escaped_cell}</th>'
            table_html += '</tr>\n</thead>\n'
            data_rows = grid[1:]
        else:
            data_rows = grid
        
        # Create body
        table_html += '<tbody>\n'
        for i, row in enumerate(data_rows):
            bg_color = '#F8F9FA' if i % 2 == 0 else 'white'
            table_html += f'<tr style="background-color: {bg_color};">'
            for cell in row:
                escaped_cell = html.escape(str(cell).strip()) if cell else ''
                table_html += f'<td style="padding: 10px 12px; border: 1px solid #ddd; vertical-align: top;">{escaped_cell}</td>'
            table_html += '</tr>\n'
        table_html += '</tbody>\n</table>\n</div>\n'
        
        return table_html
        
    except Exception as e:
        log.warning(f"Error creating HTML table: {str(e)}")
        return f'<div style="color: red; border: 1px dashed red; padding: 10px; margin: 10px 0;">Error displaying table: {html.escape(str(e))}</div>'

def create_html_figure_with_context(figure_data: dict, figure_index: int = None) -> str:
    """Create properly formatted HTML figure with better presentation."""
    try:
        caption = figure_data.get('caption', 'Untitled Figure')
        url = figure_data.get('url', '')
        content = figure_data.get('content', '')
        figure_id = figure_data.get('id', f'figure_{figure_index}')
        
        if url and url != 'placeholder://chart1' and not url.startswith('placeholder://'):
            return f'''
            <div class="figure-container" style="text-align: center; margin: 20px 0; padding: 15px; border: 1px solid #e0e0e0; border-radius: 5px; background-color: #fafafa;">
                {f'<h4 style="color: #1B5E20; margin-bottom: 10px; font-weight: bold;">Figure {figure_index}</h4>' if figure_index else ''}
                <img src="{html.escape(url)}" alt="{html.escape(caption)}" style="max-width: 100%; height: auto; border: 1px solid #ccc; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 10px;">
                <p style="font-style: italic; color: #666; margin: 5px 0; font-size: 14px;"><strong>Caption:</strong> {html.escape(caption)}</p>
                {f'<p style="color: #888; font-size: 12px; margin-top: 5px;">{html.escape(content[:200])}...</p>' if content and len(content) > 10 else ''}
            </div>
            '''
        else:
            return f'''
            <div class="figure-placeholder" style="border: 2px dashed #4CAF50; padding: 20px; text-align: center; margin: 20px 0; background-color: #F1F8E9; border-radius: 5px;">
                {f'<h4 style="color: #2E7D32; margin-bottom: 10px; font-weight: bold;">Figure {figure_index}</h4>' if figure_index else ''}
                <div style="font-size: 48px; color: #4CAF50; margin-bottom: 10px;">[CHART]</div>
                <strong style="color: #2E7D32; display: block; margin-bottom: 5px;">{html.escape(caption)}</strong>
                {f'<p style="color: #666; font-size: 12px; margin-top: 5px;">{html.escape(content[:150])}...</p>' if content else ''}
                <small style="color: #888;">Figure placeholder - original content not available</small>
            </div>
            '''
            
    except Exception as e:
        log.warning(f"Error creating HTML figure: {str(e)}")
        return f'<div style="color: red; border: 1px dashed red; padding: 10px; margin: 10px 0;">Error displaying figure: {html.escape(str(e))}</div>'

def create_html_links_section(links: List[dict]) -> str:
    """Create a properly formatted links section."""
    if not links:
        return ""
    
    links_html = '''
    <div class="links-section" style="margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-left: 4px solid #1B5E20; border-radius: 0 5px 5px 0;">
        <h4 style="color: #1B5E20; margin: 0 0 10px 0; font-weight: bold;">Related Links</h4>
        <ul style="margin: 0; padding-left: 20px;">
    '''
    
    for link in links:
        text = link.get('text', '').strip()
        url = link.get('url', '').strip()
        if text and url:
            links_html += f'<li style="margin-bottom: 5px;"><a href="{html.escape(url)}" style="color: #1565C0; text-decoration: underline;" target="_blank">{html.escape(text)}</a></li>\n'
    
    links_html += '</ul>\n</div>\n'
    return links_html

def find_content_insertion_points(content: str) -> Dict[str, List[int]]:
    """Find appropriate insertion points for tables and figures in content."""
    table_points = []
    figure_points = []
    
    # Find table references
    table_matches = re.finditer(r'(?i)(table\s*\d*|data\s*table|\|\s*\w+\s*\|)', content)
    for match in table_matches:
        table_points.append(match.end())
    
    # Find figure references
    figure_matches = re.finditer(r'(?i)(figure\s*\d*|chart|diagram|image)', content)
    for match in figure_matches:
        figure_points.append(match.end())
    
    return {'tables': table_points, 'figures': figure_points}

async def consolidate_content_with_ai(extracted_contents: List[Dict[str, Any]]) -> str:
    """
    Enhanced consolidation with proper integration of tables, figures, and hyperlinks.
    """
    
    # Start with document header
    consolidated_content = '''
    <div style="max-width: 1200px; margin: 0 auto; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333;">
        <h1 style="color: #1B5E20; text-align: center; border-bottom: 3px solid #1B5E20; padding-bottom: 10px; margin-bottom: 30px;">Consolidated Document</h1>
    '''
    
    # Process each document
    for doc_index, pdf_content in enumerate(extracted_contents, 1):
        file_name = pdf_content.get('filename', f'Document_{doc_index}')
        content = pdf_content.get('content', '')
        tables = pdf_content.get('tables', [])
        figures = pdf_content.get('figures', [])
        hyperlinks = pdf_content.get('hyperlinks', [])
        
        # Document section header
        consolidated_content += f'''
        <div class="document-section" style="margin: 30px 0; padding: 20px; border: 1px solid #e0e0e0; border-radius: 8px; background-color: #fefefe;">
            <h2 style="color: #1B5E20; border-bottom: 2px solid #4CAF50; padding-bottom: 8px; margin-bottom: 20px;">
                [FILE] Content from {html.escape(file_name)}
            </h2>
        '''
        
        # Process main content with embedded tables and figures
        if content:
            processed_content = process_content_with_embeds(content, tables, figures)
            consolidated_content += f'<div class="main-content">{processed_content}</div>\n'
        
        # Add any remaining tables not embedded in content
        remaining_tables = [t for t in tables if not is_table_embedded_in_content(t, content)]
        if remaining_tables:
            consolidated_content += '<div class="additional-tables">\n<h3 style="color: #2E7D32; margin: 20px 0 15px 0;">[TABLES] Additional Tables</h3>\n'
            for i, table in enumerate(remaining_tables, 1):
                table_html = create_html_table_with_context(table, i)
                consolidated_content += table_html
            consolidated_content += '</div>\n'
        
        # Add any remaining figures not embedded in content
        remaining_figures = [f for f in figures if not is_figure_embedded_in_content(f, content)]
        if remaining_figures:
            consolidated_content += '<div class="additional-figures">\n<h3 style="color: #2E7D32; margin: 20px 0 15px 0;">[FIGURES] Additional Figures</h3>\n'
            for i, figure in enumerate(remaining_figures, 1):
                figure_html = create_html_figure_with_context(figure, i)
                consolidated_content += figure_html
            consolidated_content += '</div>\n'
        
        # Add hyperlinks section
        if hyperlinks:
            links_html = create_html_links_section(hyperlinks)
            consolidated_content += links_html
        
        consolidated_content += '</div>\n'  # Close document section
    
    consolidated_content += '</div>\n'  # Close main container
    
    log.info(f"Enhanced consolidation completed with {len(extracted_contents)} documents")
    return consolidated_content

def process_content_with_embeds(content: str, tables: List[dict], figures: List[dict]) -> str:
    """Process content and embed tables and figures in appropriate locations."""
    
    # Convert markdown to HTML first
    processed_content = convert_markdown_to_html(content)
    
    # Find insertion points
    insertion_points = find_content_insertion_points(processed_content)
    
    # Insert tables at appropriate points
    table_insertions = 0
    for i, table in enumerate(tables):
        if i < len(insertion_points['tables']):
            insertion_pos = insertion_points['tables'][i] + table_insertions
            table_html = create_html_table_with_context(table, i + 1)
            # Insert table HTML at the position
            processed_content = (
                processed_content[:insertion_pos] + 
                f'\n{table_html}\n' + 
                processed_content[insertion_pos:]
            )
            table_insertions += len(table_html) + 2
    
    # Insert figures at appropriate points
    figure_insertions = 0
    for i, figure in enumerate(figures):
        if i < len(insertion_points['figures']):
            insertion_pos = insertion_points['figures'][i] + figure_insertions + table_insertions
            figure_html = create_html_figure_with_context(figure, i + 1)
            # Insert figure HTML at the position
            processed_content = (
                processed_content[:insertion_pos] + 
                f'\n{figure_html}\n' + 
                processed_content[insertion_pos:]
            )
            figure_insertions += len(figure_html) + 2
    
    return processed_content

def convert_markdown_to_html(content: str) -> str:
    """Convert markdown content to properly formatted HTML."""
    if not content:
        return ""
    
    # Convert headers
    content = re.sub(r'^### (.*?)$', r'<h3 style="color: #388E3C; margin: 15px 0 10px 0;">\1</h3>', content, flags=re.MULTILINE)
    content = re.sub(r'^## (.*?)$', r'<h2 style="color: #2E7D32; margin: 20px 0 15px 0;">\1</h2>', content, flags=re.MULTILINE)
    content = re.sub(r'^# (.*?)$', r'<h1 style="color: #1B5E20; margin: 25px 0 20px 0;">\1</h1>', content, flags=re.MULTILINE)
    
    # Convert lists with better formatting
    list_items = re.findall(r'^- (.*?)$', content, flags=re.MULTILINE)
    if list_items:
        # Replace list items with HTML
        for item in list_items:
            content = content.replace(f'- {item}', f'<li style="margin-bottom: 5px;">{item}</li>')
        
        # Wrap consecutive list items in ul tags
        content = re.sub(r'(<li[^>]*>.*?</li>\s*)+', r'<ul style="margin: 10px 0; padding-left: 25px;">\g<0></ul>', content, flags=re.DOTALL)
    
    # Convert paragraphs
    paragraphs = re.split(r'\n\s*\n', content)
    html_paragraphs = []
    
    for para in paragraphs:
        para = para.strip()
        if para:
            # Don't wrap if it's already HTML
            if not (para.startswith('<') and para.endswith('>')):
                # Check if it's a list or header
                if not (para.startswith('<ul') or para.startswith('<li') or para.startswith('<h')):
                    html_paragraphs.append(f'<p style="margin: 10px 0; text-align: justify; line-height: 1.6;">{para}</p>')
                else:
                    html_paragraphs.append(para)
            else:
                html_paragraphs.append(para)
    
    return '\n'.join(html_paragraphs)

def is_table_embedded_in_content(table: dict, content: str) -> bool:
    """Check if a table is already referenced/embedded in the content."""
    if not content or not table.get('grid'):
        return False
    
    grid = table.get('grid', [])
    if not grid:
        return False
    
    # Check if any cell content appears in the main content
    for row in grid[:2]:  # Check first two rows
        for cell in row:
            cell_text = str(cell).strip()
            if cell_text and len(cell_text) > 2 and cell_text.lower() in content.lower():
                return True
    return False

def is_figure_embedded_in_content(figure: dict, content: str) -> bool:
    """Check if a figure is already referenced/embedded in the content."""
    if not content:
        return False
    
    caption = figure.get('caption', '').strip()
    figure_id = figure.get('id', '').strip()
    
    # Check if caption or ID appears in content
    return (caption and len(caption) > 3 and caption.lower() in content.lower()) or \
           (figure_id and figure_id.lower() in content.lower())

def create_table_from_html(table_html: str) -> Optional[Table]:
    """Create ReportLab Table from HTML table string using BeautifulSoup for better parsing."""
    try:
        soup = BeautifulSoup(table_html, 'html.parser')
        table_tag = soup.find('table')
        if not table_tag:
            return None
        
        # Extract all rows
        rows = table_tag.find_all('tr')
        table_data = []
        
        for row in rows:
            cells = row.find_all(['th', 'td'])
            row_data = []
            for cell in cells:
                # Get text content and clean it
                text = cell.get_text(strip=True)
                # Limit cell text length to prevent overflow
                if len(text) > 100:
                    text = text[:97] + "..."
                row_data.append(text)
            if row_data:
                table_data.append(row_data)
        
        if not table_data:
            return None
        
        # Ensure all rows have the same number of columns
        max_cols = max(len(row) for row in table_data) if table_data else 0
        for row in table_data:
            while len(row) < max_cols:
                row.append("")
        
        # Calculate column widths dynamically
        available_width = 6.5 * inch
        col_widths = [available_width / max_cols] * max_cols
        
        # Create table
        table = Table(table_data, colWidths=col_widths, repeatRows=1)
        
        # Apply enhanced styling
        table_style = [
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
        
        # Check if first row contains headers
        first_row = rows[0] if rows else None
        has_header = first_row and (first_row.find('th') is not None or 
                                   'background-color: #1B5E20' in str(first_row))
        
        if has_header:
            # Header styling
            table_style.extend([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1B5E20')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
            ])
            
            # Alternating row colors for data rows
            for i in range(1, len(table_data)):
                if i % 2 == 0:
                    table_style.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#F8F9FA')))
        else:
            # No header - alternating colors for all rows
            for i in range(len(table_data)):
                if i % 2 == 0:
                    table_style.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#F8F9FA')))
        
        table.setStyle(TableStyle(table_style))
        return table
        
    except Exception as e:
        log.warning(f"Error creating table from HTML: {str(e)}")
        return None

def safe_paragraph(text: str, style) -> Paragraph:
    """Create a safe Paragraph with proper HTML handling."""
    if not text or not text.strip():
        return Spacer(1, 6)
    
    try:
        # Clean up the text and convert HTML to ReportLab markup
        text = text.strip()
        
        # Remove HTML div and styling attributes but keep content
        text = re.sub(r'<div[^>]*>', '', text)
        text = re.sub(r'</div>', '', text)
        text = re.sub(r'<span[^>]*>', '', text)
        text = re.sub(r'</span>', '', text)
        text = re.sub(r'class="[^"]*"', '', text)
        text = re.sub(r'style="[^"]*"', '', text)
        
        # Convert HTML links to ReportLab links
        text = re.sub(r'<a\s+href="([^"]*)"[^>]*>(.*?)</a>', 
                     r'<link href="\1" color="#1565C0"><u>\2</u></link>', text)
        
        # Convert HTML bold/strong tags
        text = re.sub(r'<(?:strong|b)>(.*?)</(?:strong|b)>', r'<b>\1</b>', text)
        
        # Convert HTML italic/em tags
        text = re.sub(r'<(?:em|i)>(.*?)</(?:em|i)>', r'<i>\1</i>', text)
        
        # Remove other HTML tags that ReportLab doesn't support
        text = re.sub(r'<(?!/?[biulk]|/?font|/?link)[^>]*?>', '', text)
        
        # Handle line breaks
        text = text.replace('<br/>', '<br/>')
        text = text.replace('<br>', '<br/>')
        
        # Truncate very long text
        if len(text) > 2000:
            text = text[:1997] + "..."
        
        return Paragraph(text, style)
        
    except Exception as e:
        log.warning(f"Failed to create paragraph: {str(e)}")
        # Fallback to plain text
        plain_text = re.sub(r'<[^>]+>', '', text)
        plain_text = html.escape(plain_text)
        if len(plain_text) > 1000:
            plain_text = plain_text[:997] + "..."
        return Paragraph(plain_text, style)

def html_to_pdf(html_content: str, output_path: str, temp_dir: str = None):
    """Convert HTML content to PDF using ReportLab with improved parsing."""
    try:
        print(f"DEBUG: Starting PDF generation to: {output_path}")
        
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
        
        # Create enhanced custom styles
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
        
        print(f"DEBUG: Processing HTML content with BeautifulSoup...")
        
        # Use BeautifulSoup for better HTML parsing
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style tags
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Process the content systematically
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'div', 'ul', 'li', 'table']):
            try:
                # Handle headings
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
                
                # Handle paragraphs
                elif element.name == 'p':
                    text_content = str(element)
                    if text_content.strip() and element.get_text(strip=True):
                        story.append(safe_paragraph(text_content, normal_style))
                
                # Handle tables
                elif element.name == 'table' and element.parent.name != 'div':
                    table = create_table_from_html(str(element))
                    if table:
                        story.append(Spacer(1, 8))
                        story.append(table)
                        story.append(Spacer(1, 8))
                
                # Handle special divs
                elif element.name == 'div':
                    classes = element.get('class', [])
                    
                    if 'table-container' in classes:
                        # Handle table containers
                        table_elem = element.find('table')
                        if table_elem:
                            # Add table title if present
                            title_elem = element.find('h4')
                            if title_elem:
                                story.append(safe_paragraph(title_elem.get_text(strip=True), heading3_style))
                            
                            table = create_table_from_html(str(table_elem))
                            if table:
                                story.append(table)
                                story.append(Spacer(1, 10))
                    
                    elif 'figure-container' in classes or 'figure-placeholder' in classes:
                        # Handle figure placeholders
                        title_elem = element.find('h4')
                        if title_elem:
                            story.append(safe_paragraph(title_elem.get_text(strip=True), heading3_style))
                        
                        # Get the figure description
                        text = element.get_text(separator=' ', strip=True)
                        if text and not text.startswith('Figure'):
                            # Remove the title part if it exists
                            text_parts = text.split('\n')
                            if len(text_parts) > 1:
                                text = '\n'.join(text_parts[1:]).strip()
                            story.append(safe_paragraph(text, normal_style))
                        story.append(Spacer(1, 8))
                    
                    elif 'links-section' in classes:
                        # Handle links sections
                        title_elem = element.find('h4')
                        if title_elem:
                            story.append(safe_paragraph(title_elem.get_text(strip=True), heading3_style))
                        
                        ul_elem = element.find('ul')
                        if ul_elem:
                            for li in ul_elem.find_all('li'):
                                story.append(safe_paragraph(str(li), normal_style))
                        story.append(Spacer(1, 8))
                
                # Handle lists
                elif element.name in ['ul', 'li']:
                    if element.name == 'li' and element.parent.name == 'ul':
                        continue  # Skip individual li elements, process them with ul
                    
                    if element.name == 'ul':
                        for li in element.find_all('li', recursive=False):
                            story.append(safe_paragraph(str(li), normal_style))
            
            except Exception as e:
                log.warning(f"Error processing element {element.name}: {str(e)}")
                continue
        
        # If story is still empty, add fallback content
        if not story:
            print("DEBUG: No content processed, adding fallback")
            story.append(safe_paragraph("Document processed successfully, but no readable content was found.", normal_style))
        
        print(f"DEBUG: Content processing completed. Story has {len(story)} elements")
        
        # Build PDF
        doc.build(story)
        
        # Verify file was created
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"DEBUG: PDF created successfully. Size: {file_size} bytes")
            if file_size == 0:
                raise Exception("PDF file was created but is empty")
        else:
            raise Exception("PDF file was not created")
            
        log.info(f"PDF generated successfully at {output_path}")
        
    except Exception as e:
        print(f"DEBUG: Error in PDF generation: {str(e)}")
        log.error(f"Error converting HTML to PDF: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")
# --- Imports ---
import os
import re
import json
import difflib
import pandas as pd
import fitz  # PyMuPDF
from collections import OrderedDict
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Utility Functions (kept for future use) ---
def normalize_paragraph(text):
    """Lowercase, strip, and remove leading bullets/specials from paragraph content."""
    text = text.lower().strip()
    # Replace unicode ellipsis with three dots
    text = text.replace('ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦', '...')
    # Remove all leading punctuation, ellipsis, and whitespace
    text = re.sub(r'^[\s\.,;:Ãƒâ€šÃ‚Â·ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢\-\*]+', '', text)
    # Remove repeated dots at the start (e.g. ... or ....)
    text = re.sub(r'^\.+', '', text)
    # Remove any remaining leading non-alphanumeric chars
    text = re.sub(r'^[^\w\d]+', '', text)
    return text

def calc_similarity(a, b):
    """Return SequenceMatcher similarity ratio between two strings."""
    return difflib.SequenceMatcher(None, a, b).ratio()

def is_rect_overlapping(rect1, rect2, threshold=0.5):
    """Check if two rectangles (fitz.Rect) overlap significantly."""
    intersect = fitz.Rect(rect1)
    intersect.intersect(rect2)
    
    if intersect.is_empty:
        return False
        
    intersect_area = intersect.width * intersect.height
    # Check overlap against the smaller of the two rects to handle containment
    area1 = rect1.width * rect1.height
    area2 = rect2.width * rect2.height
    
    # Avoid division by zero for empty rects
    min_area = min(area1, area2)
    if min_area == 0:
        return False

    if (intersect_area / min_area) > threshold:
        return True
        
    return False

# --- Table Exclusion Utilities ---
def para_in_table_by_elements(idx, table_indices):
    """Check if a paragraph is referenced by any table (element reference)."""
    result = idx in table_indices if table_indices else False
    return result

def para_in_table_by_overlap(para, idx, tables):
    """Check if a paragraph spatially overlaps with any table (multi-span support)."""
    if not tables:
        return False
    try:
        para_page = para.get('boundingRegions', [{}])[0].get('pageNumber', None)
        para_spans = para.get('spans', [])
        if para_page is None or not para_spans:
            return False
        for para_span in para_spans:
            para_start = para_span.get('offset', None)
            para_len = para_span.get('length', None)
            if para_start is None or para_len is None:
                continue
            para_end = para_start + para_len
            for table in tables:
                t_page = table.get('boundingRegions', [{}])[0].get('pageNumber', None)
                for t_span in table.get('spans', []):
                    t_start = t_span.get('offset', None)
                    t_len = t_span.get('length', None)
                    if t_page is None or t_start is None or t_len is None:
                        continue
                    t_end = t_start + t_len
                    if t_page != para_page:
                        continue
                    if (para_start >= t_start and para_start < t_end) or \
                       (para_end > t_start and para_end <= t_end) or \
                       (para_start <= t_start and para_end >= t_end):
                        return True
        return False
    except Exception as e:
        print(f"[DEBUG] Exception in para_in_table_by_overlap for idx {idx}: {e}")
        return False

def should_skip_paragraph(idx, para, tables, table_indices):
    """Determine if a paragraph should be skipped (table content, footnote, caption)."""
    if para_in_table_by_elements(idx, table_indices):
        return True
    if para_in_table_by_overlap(para, idx, tables):
        return True
    # Skip paragraphs referenced by table footnotes
    for table in tables or []:
        for footnote in table.get('footnotes', []):
            for el in footnote.get('elements', []):
                if isinstance(el, str) and el.startswith('/paragraphs/'):
                    try:
                        foot_idx = int(el.split('/')[-1])
                        if idx == foot_idx:
                            return True
                    except Exception:
                        continue
        # Skip paragraphs referenced by table captions
        caption = table.get('caption', {})
        if caption:
            for el in caption.get('elements', []):
                if isinstance(el, str) and el.startswith('/paragraphs/'):
                    try:
                        caption_idx = int(el.split('/')[-1])
                        if idx == caption_idx:
                            return True
                    except Exception:
                        continue
    return False

# --- Document Intelligence credentials from .env ---
DOCUMENTINTELLIGENCE_API_KEY = os.getenv("DOCUMENTINTELLIGENCE_API_KEY")
DOCUMENTINTELLIGENCE_ENDPOINT = os.getenv("DOCUMENTINTELLIGENCE_ENDPOINT")

# --- Setup paths ---
pdf_folder = r"C:\Users\XJ533JH\Downloads\doc_ext\input_pdfs"
output_folder = r"C:\Users\XJ533JH\Downloads\doc_ext\extracted_content"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Initialize Document Intelligence client
document_intelligence_client = DocumentIntelligenceClient(
    endpoint=DOCUMENTINTELLIGENCE_ENDPOINT, 
    credential=AzureKeyCredential(DOCUMENTINTELLIGENCE_API_KEY)
)

def extract_content_by_headings(paragraphs, all_tables, table_paragraph_indices):
    """Extract content organized by section headings."""
    content_structure = {}
    heading_counter = 1
    current_heading = None
    current_content = []
    skip_until_idx = -1  # Track paragraphs to skip due to Note processing
    
    for idx, para in enumerate(paragraphs):
        # Skip if we've already processed this paragraph during Note look-ahead
        if idx <= skip_until_idx:
            continue
            
        # Skip table content
        if should_skip_paragraph(idx, para, all_tables, table_paragraph_indices):
            continue
        
        # Skip footers, headers, and page numbers
        para_role = para.get('role', '').lower()
        if para_role in ["pagefooter", "pageheader", "pagenumber"]:
            continue
            
        para_content = para.get('content', '').strip()
        
        # Check if this is a section heading or title
        if para_role in ("sectionheading", "title"):
            # Special case: if heading is "Note", combine it with the content that follows
            if para_content.lower() == "note":
                # Look ahead to get the content that follows the Note heading
                note_content = []
                next_idx = idx + 1
                while next_idx < len(paragraphs):
                    next_para = paragraphs[next_idx]
                    # Skip table content
                    if should_skip_paragraph(next_idx, next_para, all_tables, table_paragraph_indices):
                        next_idx += 1
                        continue
                    # Skip footers, headers, and page numbers
                    next_para_role = next_para.get('role', '').lower()
                    if next_para_role in ["pagefooter", "pageheader", "pagenumber"]:
                        next_idx += 1
                        continue
                    # Stop if we hit another section heading
                    if next_para_role in ("sectionheading", "title"):
                        break
                    # Add content
                    next_content = next_para.get('content', '').strip()
                    if next_content:
                        note_content.append(next_content)
                    next_idx += 1
                
                # Add "Note" line followed by the collected content to current heading
                current_content.append("Note")
                if note_content:
                    current_content.extend(note_content)
                
                # Set skip_until_idx to avoid reprocessing the same content
                skip_until_idx = next_idx - 1
                continue
            else:
                # Save previous heading and content if exists
                if current_heading is not None:
                    content_structure[f"heading_{heading_counter:03d}"] = {
                        "heading": current_heading,
                        "combined_content": '\n'.join(current_content).strip()
                    }
                    heading_counter += 1
                
                # Start new heading
                current_heading = para_content
                current_content = []
        else:
            # Add content under current heading
            if para_content:  # Only add non-empty content
                # Special case: if content starts with "Note:", include it in the same paragraph
                if para_content.lower().startswith("note:"):
                    if current_content:
                        current_content[-1] += " " + para_content
                    else:
                        current_content.append(para_content)
                else:
                    current_content.append(para_content)
    
    # Save the last heading and content
    if current_heading is not None:
        content_structure[f"heading_{heading_counter:03d}"] = {
            "heading": current_heading,
            "combined_content": '\n'.join(current_content).strip()
        }
    
    return content_structure

def extract_tables_to_excel(all_tables, excel_path):
    """Extract tables and save each table as a separate sheet in Excel."""
    if not all_tables:
        print("No tables found in the document.")
        return
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for table_idx, table in enumerate(all_tables, 1):
            # Get table cells
            cells = table.get('cells', [])
            if not cells:
                continue
                
            # Find max row and column indices for actual table data
            max_row = max(cell.get('rowIndex', 0) for cell in cells) + 1
            max_col = max(cell.get('columnIndex', 0) for cell in cells) + 1
            
            # Prepare the final data list
            final_data = []
            
            # 1. Initialize table grid for actual table data (headers + content)
            table_grid = [[''] * max_col for _ in range(max_row)]
            
            # Fill the grid with cell content
            for cell in cells:
                row_idx = cell.get('rowIndex', 0)
                col_idx = cell.get('columnIndex', 0)
                content = cell.get('content', '').strip()
                table_grid[row_idx][col_idx] = content
            
            # 2. Add table data (headers + content) to final data
            final_data.extend(table_grid)
            
            # 3. Add 3 empty rows gap
            for _ in range(3):
                final_data.append([''] * max_col)
            
            # 4. Add CAPTION section (always present)
            caption = table.get('caption', {})
            # Add "CAPTION" heading
            caption_heading_row = ['CAPTION'] + [''] * (max_col - 1)
            final_data.append(caption_heading_row)
            # Add actual caption content if exists, otherwise empty row
            if caption and 'content' in caption:
                caption_content_row = [caption['content']] + [''] * (max_col - 1)
                final_data.append(caption_content_row)
            else:
                final_data.append([''] * max_col)  # Empty row if no caption
            final_data.append([''] * max_col)  # Empty row after caption
            
            # 5. Add FOOTNOTES section (always present)
            footnotes = table.get('footnotes', [])
            # Add "FOOTNOTES" heading
            footnotes_heading_row = ['FOOTNOTES'] + [''] * (max_col - 1)
            final_data.append(footnotes_heading_row)
            # Add actual footnote content if exists, otherwise empty row
            if footnotes:
                for footnote in footnotes:
                    if 'content' in footnote:
                        footnote_content_row = [footnote['content']] + [''] * (max_col - 1)
                        final_data.append(footnote_content_row)
            else:
                final_data.append([''] * max_col)  # Empty row if no footnotes
            
            # Convert to DataFrame
            df = pd.DataFrame(final_data)
            
            # Create sheet name
            sheet_name = f"Table_{table_idx}"
            
            # Write to Excel
            df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
            print(f"Saved table {table_idx} to sheet: {sheet_name}")
            
            # Print what was added
            print(f"  - Added CAPTION section" + (f": {caption['content'][:50]}..." if caption and 'content' in caption else " (empty)"))
            print(f"  - Added FOOTNOTES section" + (f" with {len(footnotes)} footnote(s)" if footnotes else " (empty)"))

def get_table_headers(table):
    """Extract column headers from a table."""
    headers = []
    cells = table.get('cells', [])
    
    # Find all cells in row 0 (header row)
    header_cells = [cell for cell in cells if cell.get('rowIndex') == 0]
    # Sort by column index
    header_cells.sort(key=lambda x: x.get('columnIndex', 0))
    
    for cell in header_cells:
        headers.append(cell.get('content', '').strip())
    
    return headers

def get_table_page_number(table):
    """Get the page number of a table."""
    bounding_regions = table.get('boundingRegions', [])
    if bounding_regions:
        return bounding_regions[0].get('pageNumber', 0)
    return 0

def are_headers_similar(headers1, headers2, similarity_threshold=0.8):
    """Check if two header lists are similar enough to be considered the same table."""
    if len(headers1) != len(headers2):
        return False
    
    if not headers1 or not headers2:
        return False
    
    # Check exact match first
    if headers1 == headers2:
        return True
    
    # Check similarity for each header pair
    matches = 0
    for h1, h2 in zip(headers1, headers2):
        if h1.lower() == h2.lower():
            matches += 1
        elif h1 and h2 and (h1.lower() in h2.lower() or h2.lower() in h1.lower()):
            matches += 1
    
    similarity = matches / len(headers1)
    return similarity >= similarity_threshold

def combine_split_tables(tables):
    """Combine tables that are split across multiple pages."""
    if not tables:
        return tables
    
    print(f"Processing {len(tables)} tables for potential combination...")
    
    # Sort tables by page number
    tables_with_pages = []
    for i, table in enumerate(tables):
        page_num = get_table_page_number(table)
        tables_with_pages.append((page_num, i, table))
    
    tables_with_pages.sort(key=lambda x: x[0])  # Sort by page number
    
    combined_tables = []
    skip_indices = set()
    
    i = 0
    while i < len(tables_with_pages):
        if i in skip_indices:
            i += 1
            continue
            
        current_page, current_idx, current_table = tables_with_pages[i]
        current_headers = get_table_headers(current_table)
        
        # Look for tables on subsequent pages with similar headers
        tables_to_combine = [current_table]
        last_page = current_page
        
        j = i + 1
        while j < len(tables_with_pages):
            next_page, next_idx, next_table = tables_with_pages[j]
            next_headers = get_table_headers(next_table)
            
            # Check if it's on the next page and has similar headers
            if (next_page == last_page + 1 and 
                are_headers_similar(current_headers, next_headers)):
                
                tables_to_combine.append(next_table)
                skip_indices.add(j)
                last_page = next_page
                print(f"Found continuation: Table on page {current_page} continues on page {next_page}")
                j += 1
            else:
                break
        
        # Combine the tables if we found continuations
        if len(tables_to_combine) > 1:
            combined_table = combine_table_parts(tables_to_combine)
            combined_tables.append(combined_table)
            print(f"Combined {len(tables_to_combine)} table parts from pages {current_page} to {last_page}")
        else:
            combined_tables.append(current_table)
        
        i += 1
    
    print(f"Resulted in {len(combined_tables)} tables after combination.")
    return combined_tables

def combine_table_parts(table_parts):
    """Combine multiple table parts into a single table."""
    if len(table_parts) == 1:
        return table_parts[0]
    
    # Use the first table as the base
    combined_table = table_parts[0].copy()
    combined_cells = list(combined_table.get('cells', []))
    
    # Find the maximum row index in the first table
    max_row_idx = max(cell.get('rowIndex', 0) for cell in combined_cells) if combined_cells else -1
    
    # Add cells from subsequent tables, adjusting row indices
    for table_part in table_parts[1:]:
        cells = table_part.get('cells', [])
        
        # Skip header row (row 0) in continuation tables
        non_header_cells = [cell for cell in cells if cell.get('rowIndex', 0) > 0]
        
        for cell in non_header_cells:
            new_cell = cell.copy()
            # Adjust row index to continue from where the previous table ended
            new_cell['rowIndex'] = cell.get('rowIndex', 0) + max_row_idx
            combined_cells.append(new_cell)
        
        # Update max_row_idx for the next iteration
        if non_header_cells:
            max_row_idx = max(cell['rowIndex'] for cell in combined_cells)
    
    # Update the combined table
    combined_table['cells'] = combined_cells
    combined_table['rowCount'] = max_row_idx + 1
    
    # Combine bounding regions from all parts
    all_bounding_regions = []
    for table_part in table_parts:
        all_bounding_regions.extend(table_part.get('boundingRegions', []))
    combined_table['boundingRegions'] = all_bounding_regions
    
    # Combine spans from all parts
    all_spans = []
    for table_part in table_parts:
        all_spans.extend(table_part.get('spans', []))
    combined_table['spans'] = all_spans
    
    return combined_table

def build_table_paragraph_index(tables):
    """Build set of all paragraph indices referenced by tables."""
    indices = set()
    for table in tables:
        for el in table.get('elements', []):
            if isinstance(el, str) and el.startswith('/paragraphs/'):
                try:
                    idx = int(el.split('/')[-1])
                    indices.add(idx)
                except Exception:
                    continue
    return indices

def calculate_iou(rect1, rect2):
    """Calculate the Intersection over Union (IoU) of two rectangles (fitz.Rect)."""
    # Determine the coordinates of the intersection rectangle
    x_left = max(rect1.x0, rect2.x0)
    y_top = max(rect1.y0, rect2.y0)
    x_right = min(rect1.x1, rect2.x1)
    y_bottom = min(rect1.y1, rect2.y1)

    # If there is no overlap, return 0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The area of intersection
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # The area of both rectangles
    rect1_area = rect1.width * rect1.height
    rect2_area = rect2.width * rect2.height
    
    # The area of their union
    union_area = rect1_area + rect2_area - intersection_area
    
    if union_area == 0:
        return 0.0
        
    iou = intersection_area / union_area
    return iou

def process_pdf(pdf_name, pdf_folder, output_folder):
    """Process a single PDF file."""
    pdf_path = os.path.join(pdf_folder, pdf_name)
    if not os.path.isfile(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    print(f"Processing: {pdf_name}")
    try:
        # Get base filename without extension
        pdf_base = os.path.splitext(pdf_name)[0]
        
        # Create subfolder for this PDF
        pdf_output_folder = os.path.join(output_folder, pdf_base)
        os.makedirs(pdf_output_folder, exist_ok=True)
        
        # Create an images subfolder
        images_folder = os.path.join(pdf_output_folder, "images")
        os.makedirs(images_folder, exist_ok=True)
        
        # Analyze document with Document Intelligence
        with open(pdf_path, "rb") as f:
            poller = document_intelligence_client.begin_analyze_document(
                "prebuilt-layout", f, content_type="application/octet-stream"
            )
            result = poller.result()

        # --- New: Get all paragraph bounding boxes for image de-duplication ---
        all_para_rects_by_page = {}
        if result.paragraphs:
            for para in result.paragraphs:
                if not para.bounding_regions:
                    continue
                region = para.bounding_regions[0]
                page_num = region.page_number
                
                polygon_coords_inches = region.polygon
                if not polygon_coords_inches:
                    continue

                # Convert from inches to points (1 inch = 72 points)
                polygon_coords_points = [p * 72 for p in polygon_coords_inches]
                x_coords = polygon_coords_points[0::2]
                y_coords = polygon_coords_points[1::2]
                
                if not x_coords or not y_coords:
                    continue
                
                para_rect = fitz.Rect(min(x_coords), min(y_coords), max(x_coords), max(y_coords))
                
                if page_num not in all_para_rects_by_page:
                    all_para_rects_by_page[page_num] = []
                all_para_rects_by_page[page_num].append(para_rect)
        
        # --- Unified Content Extraction ---
        
        # Open the PDF with PyMuPDF to extract images
        pdf_document = fitz.open(pdf_path)
        
        # Create a list to hold all content elements
        all_content = []

        # --- New: Identify all character offsets occupied by tables ---
        table_char_offsets = set()
        if result.tables:
            for table in result.tables:
                for span in table.spans:
                    for i in range(span.offset, span.offset + span.length):
                        table_char_offsets.add(i)

        # 1. Add paragraphs to the list, skipping any that are part of a table
        if result.paragraphs:
            for para in result.paragraphs:
                # A paragraph is considered part of a table if its first span starts inside a table's character range.
                is_part_of_table = False
                if para.spans and table_char_offsets:
                    # Check if the beginning of the paragraph is within a table's character span
                    if para.spans[0].offset in table_char_offsets:
                        is_part_of_table = True
                
                if is_part_of_table:
                    continue

                # --- New: Skip paragraphs that only contain branding words ---
                cleaned_content = para.content.strip()
                # Using lower() for case-insensitive comparison
                lower_content = cleaned_content.lower()
                branding_keywords = ['enbridge', 'cÃ©nbridgeÂ®', 'dominion energy', 'dominion energyÂ®']
                if lower_content in branding_keywords:
                    print(f"  - Skipping isolated branding paragraph: '{cleaned_content}'")
                    continue

                all_content.append({
                    "type": "paragraph",
                    "role": para.role,
                    "content": para.content,
                    "bounding_regions": [{"page_number": region.page_number, "polygon": region.polygon} for region in para.bounding_regions],
                    "spans": [{"offset": span.offset, "length": span.length} for span in para.spans]
                })
            
        # 2. Add tables to the list
        if result.tables:
            for i, table in enumerate(result.tables):
                table_data = []
                row_count = table.row_count if table.row_count is not None else 0
                col_count = table.column_count if table.column_count is not None else 0
                cells_by_coords = {(cell.row_index, cell.column_index): cell for cell in table.cells}
                
                for r in range(row_count):
                    row_data = []
                    for c in range(col_count):
                        cell = cells_by_coords.get((r, c))
                        row_data.append(cell.content if cell and cell.content else "")
                    table_data.append(row_data)

                all_content.append({
                    "type": "table",
                    "id": f"table_{i}",
                    "bounding_regions": [{"page_number": region.page_number, "polygon": region.polygon} for region in table.bounding_regions],
                    "spans": [{"offset": span.offset, "length": span.length} for span in table.spans],
                    "data": table_data
                })

        # --- De-duplication logic for images ---
        processed_figure_rects = []

        # 3. Extract and add figures (images) from Document Intelligence
        if result.figures:
            print(f"Found {len(result.figures)} figures to process from Document Intelligence.")
            for i, figure in enumerate(result.figures):
                if not figure.bounding_regions:
                    print(f"  - Skipping figure {i+1} (no bounding regions).")
                    continue
                
                region = figure.bounding_regions[0]
                page_num = region.page_number
                page = pdf_document[page_num - 1]
                
                polygon_coords_inches = region.polygon
                if not polygon_coords_inches:
                    print(f"  - Skipping figure {i+1} on page {page_num} (no polygon coordinates).")
                    continue

                polygon_coords_points = [p * 72 for p in polygon_coords_inches]
                x_coords = polygon_coords_points[0::2]
                y_coords = polygon_coords_points[1::2]

                if not x_coords or not y_coords:
                    print(f"  - Skipping figure {i+1} on page {page_num} (invalid polygon coordinates).")
                    continue
                
                clip_rect = fitz.Rect(min(x_coords), min(y_coords), max(x_coords), max(y_coords))
                
                if clip_rect.is_empty or clip_rect.width < 2 or clip_rect.height < 2:
                    print(f"  - Skipping figure {i+1} on page {page_num} (invalid or too small rectangle: {clip_rect}).")
                    continue

                # Filter out small decorative images
                if clip_rect.width < 40 and clip_rect.height < 40:
                    print(f"  - Skipping figure {i+1} on page {page_num} (likely decorative, size: {clip_rect.width:.1f}x{clip_rect.height:.1f}).")
                    continue

                # --- New: Check for logos by looking at nearby text ---
                is_logo = False
                # Find the corresponding paragraph for the figure to check for branding keywords
                for para in result.paragraphs:
                    if not para.bounding_regions or para.bounding_regions[0].page_number != page_num:
                        continue

                    para_region = para.bounding_regions[0]
                    para_polygon_points = [p * 72 for p in para_region.polygon]
                    para_x_coords = para_polygon_points[0::2]
                    para_y_coords = para_polygon_points[1::2]
                    if not para_x_coords or not para_y_coords:
                        continue
                    para_rect = fitz.Rect(min(para_x_coords), min(para_y_coords), max(para_x_coords), max(para_y_coords))

                    # Check if the paragraph is vertically aligned and close to the figure
                    # A logo's text is often right below or above it.
                    vertical_distance = min(abs(clip_rect.y1 - para_rect.y0), abs(para_rect.y1 - clip_rect.y0))
                    horizontal_overlap = max(0, min(clip_rect.x1, para_rect.x1) - max(clip_rect.x0, para_rect.x0))
                    
                    if vertical_distance < 50 and horizontal_overlap > 0:
                        lower_content = para.content.strip().lower()
                        branding_keywords = ['enbridge', 'cÃ©nbridgeÂ®', 'dominion energy', 'dominion energyÂ®']
                        if any(keyword in lower_content for keyword in branding_keywords):
                            is_logo = True
                            print(f"  - Skipping figure {i+1} on page {page_num} (likely a logo, near text: '{para.content.strip()}').")
                            break
                if is_logo:
                    continue

                # Store the processed rectangle for de-duplication
                processed_figure_rects.append((page_num, clip_rect))

                try:
                    pix = page.get_pixmap(clip=clip_rect, colorspace=fitz.csRGB, alpha=False, dpi=300)
                    image_path = os.path.join(images_folder, f"figure_{i+1}.jpg")
                    pix.save(image_path, jpg_quality=85)
                    print(f"  - Saved figure {i+1} to {image_path}")
                    
                    width_pts = clip_rect.width
                    height_pts = clip_rect.height

                    all_content.append({
                        "type": "image",
                        "id": f"figure_{i+1}",
                        "path": os.path.relpath(image_path, pdf_output_folder).replace('\\', '/'),
                        "width_pts": width_pts,
                        "height_pts": height_pts,
                        "bounding_regions": [{"page_number": region.page_number, "polygon": polygon_coords_inches}],
                        "spans": [{"offset": span.offset, "length": span.length} for span in figure.spans]
                    })
                except Exception as e:
                    print(f"  - Error extracting figure {i+1} on page {page_num} with get_pixmap: {e}")

        # 4. Extract vector graphics / drawings from PyMuPDF, avoiding duplicates
        print("Scanning for additional vector graphics...")
        for page_idx, page in enumerate(pdf_document):
            page_num = page_idx + 1
            drawings = page.get_drawings()
            for item in drawings:
                drawing_rect = fitz.Rect(item['rect'])

                # Skip very thin lines (borders/underlines) or small decorative drawings
                if drawing_rect.is_empty or drawing_rect.width < 2 or drawing_rect.height < 2:
                    continue
                if drawing_rect.width < 40 and drawing_rect.height < 40:
                    print(f"  - Skipping drawing on page {page_num} (likely decorative, size: {drawing_rect.width:.1f}x{drawing_rect.height:.1f}).")
                    continue

                # --- De-duplication Logic for Text Backgrounds and Duplicates ---

                # 1. Check if the drawing is a background for a text paragraph.
                is_background_for_text = False
                para_rects_on_page = all_para_rects_by_page.get(page_num, [])
                
                if para_rects_on_page:
                    for para_rect in para_rects_on_page:
                        intersect_rect = fitz.Rect(drawing_rect)
                        intersect_rect.intersect(para_rect)
                        
                        if intersect_rect.is_empty:
                            continue

                        intersection_area = intersect_rect.width * intersect_rect.height
                        para_area = para_rect.width * para_rect.height
                        drawing_area = drawing_rect.width * drawing_rect.height

                        if (para_area > 0 and (intersection_area / para_area) > 0.85) or \
                           (drawing_area > 0 and (intersection_area / drawing_area) > 0.85):
                            is_background_for_text = True
                            break
                
                if is_background_for_text:
                    # print(f"  - Skipping drawing on page {page_num}, it's a background for text.")
                    continue

                # 1.5. Check if the drawing is a simple, solid-colored rectangle with no text overlap.
                # This targets decorative boxes/lines that are not backgrounds.
                is_isolated_solid_box = False
                if item['fill'] is not None: # It's a filled shape
                    # Check if it overlaps with ANY paragraph on the page
                    overlaps_with_any_text = False
                    if para_rects_on_page:
                        for para_rect in para_rects_on_page:
                            if not fitz.Rect(drawing_rect).intersect(para_rect).is_empty:
                                overlaps_with_any_text = True
                                break
                    
                    if not overlaps_with_any_text:
                        is_isolated_solid_box = True

                if is_isolated_solid_box:
                    print(f"  - Skipping drawing on page {page_num}, it's an isolated solid box.")
                    continue

                # 2. Check if the drawing overlaps with an already processed figure.
                is_duplicate = False
                for fig_page_num, fig_rect in processed_figure_rects:
                    if page_num == fig_page_num and is_rect_overlapping(drawing_rect, fig_rect, threshold=0.5):
                        is_duplicate = True
                        # print(f"  - Skipping drawing on page {page_num}, it overlaps with an already processed figure.")
                        break
                
                if is_duplicate:
                    continue

                print(f"  - Found new vector graphic on page {page_num} at {drawing_rect}")
                try:
                    pix = page.get_pixmap(clip=drawing_rect, colorspace=fitz.csRGB, alpha=False, dpi=300)
                    image_name = f"drawing_{page_num}_{int(drawing_rect.x0)}_{int(drawing_rect.y0)}.jpg"
                    image_path = os.path.join(images_folder, image_name)
                    pix.save(image_path, jpg_quality=95)
                    
                    width_pts = drawing_rect.width
                    height_pts = drawing_rect.height

                    all_content.append({
                        "type": "image",
                        "id": image_name.replace('.jpg', ''),
                        "path": os.path.relpath(image_path, pdf_output_folder).replace('\\', '/'),
                        "width_pts": width_pts,
                        "height_pts": height_pts,
                        "bounding_regions": [{"page_number": page_num, "polygon": list(drawing_rect)}],
                        "spans": []
                    })
                    
                    processed_figure_rects.append((page_num, drawing_rect))
                except Exception as e:
                    print(f"  - Error extracting drawing on page {page_num}: {e}")

        # Sort all content by page number and then by vertical position (y-coordinate)
        def sort_key(element):
            if not element.get("bounding_regions"):
                return (0, 0)
            br = element["bounding_regions"][0]
            page = br["page_number"]
            y_coord = br["polygon"][1] if br.get("polygon") and len(br["polygon"]) > 1 else 0
            return (page, y_coord)

        all_content.sort(key=sort_key)
        
        # Save the unified content structure to a new JSON file
        unified_json_path = os.path.join(pdf_output_folder, f"{pdf_base}_unified.json")
        with open(unified_json_path, "w", encoding="utf-8") as f:
            json.dump(all_content, f, ensure_ascii=False, indent=2)
        print(f"Saved unified content to: {unified_json_path}")
        
        pdf_document.close()

        # --- Keep existing table extraction to Excel for now ---
        all_tables_as_dict = []
        if result.tables:
            for table in result.tables:
                table_dict = {
                    "cells": [{"row_index": cell.row_index, "column_index": cell.column_index, "content": cell.content} for cell in table.cells],
                    "bounding_regions": [{"page_number": region.page_number, "polygon": region.polygon} for region in table.bounding_regions],
                    "spans": [{"offset": span.offset, "length": span.length} for span in table.spans],
                    "row_count": table.row_count,
                    "column_count": table.column_count
                }
                if hasattr(table, 'caption') and table.caption:
                    table_dict['caption'] = {'content': table.caption.content}
                if hasattr(table, 'footnotes') and table.footnotes:
                    table_dict['footnotes'] = [{'content': fn.content} for fn in table.footnotes]
                all_tables_as_dict.append(table_dict)

        combined_tables = combine_split_tables(all_tables_as_dict)
        if combined_tables:
            excel_path = os.path.join(pdf_output_folder, f"{pdf_base}.xlsx")
            extract_tables_to_excel(combined_tables, excel_path)
            print(f"Saved tables to: {excel_path}")
        else:
            print(f"No tables found in {pdf_name}")
            
    except Exception as e:
        print(f"Error processing {pdf_name}: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to process all PDFs in the input folder."""
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    for pdf_name in pdf_files:
        process_pdf(pdf_name, pdf_folder, output_folder)

if __name__ == "__main__":
    main()

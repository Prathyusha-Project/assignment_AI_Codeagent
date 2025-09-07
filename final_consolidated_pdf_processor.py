# --- Imports ---
import os
import re
import json
import difflib
import pandas as pd
from collections import OrderedDict
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def normalize_heading(text):
    """Normalize heading text for exact matching - based on backup2.py normalize_paragraph function."""
    text = text.lower().strip()
    # Replace unicode ellipsis with three dots
    text = text.replace('â€¦', '...')
    # Remove all leading punctuation, ellipsis, and whitespace
    text = re.sub(r'^[\s\.,;:Â·â€¢\-\*]+', '', text)
    # Remove repeated dots at the start (e.g. ... or ....)
    text = re.sub(r'^\.+', '', text)
    # Remove any remaining leading non-alphanumeric chars
    text = re.sub(r'^[^\w\d]+', '', text)
    return text

def extract_section_number(heading_text):
    """Extract the main section number from a heading (e.g., '4.1.2' -> '4', '1 Purpose' -> '1')."""
    import re
    # Match patterns like "1", "4.1", "4.1.1", etc. at the start of the heading
    match = re.match(r'^(\d+)', heading_text.strip())
    if match:
        return match.group(1)
    return None

def group_headings_by_section(headings_list, pdf_name):
    """Group headings by their main section number, preserving order."""
    sections = {}
    other_headings = []
    
    for heading_data in headings_list:
        heading_text = heading_data['heading']
        section_num = extract_section_number(heading_text)
        
        if section_num:
            if section_num not in sections:
                sections[section_num] = []
            sections[section_num].append(heading_data)
        else:
            # Handle special sections like "Appendix", "Note", etc.
            other_headings.append(heading_data)
    
    return sections, other_headings

def combine_text_content_from_pdfs(json_file_paths, output_path):
    """Combine text content from multiple PDF JSON files using hierarchical section-aware matching."""
    if not json_file_paths:
        print("No JSON files provided for combining.")
        return
    
    print(f"Combining text content from {len(json_file_paths)} JSON files...")
    
    # Load all content structures
    all_content_structures = []
    for json_path in json_file_paths:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                # Extract PDF name from filename (without .json extension)
                pdf_name = os.path.splitext(os.path.basename(json_path))[0]
                
                # Convert content to ordered list of headings
                headings_list = []
                for heading_key in sorted(content.keys()):  # Sort to maintain order
                    heading_data = content[heading_key]
                    original_heading = heading_data.get('heading', '').strip()
                    content_text = heading_data.get('combined_content', '').strip()
                    
                    if original_heading:  # Only need heading to exist, content can be empty
                        headings_list.append({
                            'heading': original_heading,
                            'content': content_text  # Keep even if empty
                        })
                
                # Group headings by section
                sections, other_headings = group_headings_by_section(headings_list, pdf_name)
                
                all_content_structures.append({
                    'pdf_name': pdf_name,
                    'sections': sections,
                    'other_headings': other_headings,
                    'total_headings': len(headings_list)
                })
                print(f"Loaded: {pdf_name} ({len(headings_list)} headings, {len(sections)} sections)")
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
            continue
    
    if not all_content_structures:
        print("No valid content structures loaded.")
        return
    
    # Use ordered list to maintain sequential order
    final_combined_ordered = []
    
    # Step 1: Find all unique section numbers across all PDFs
    all_section_numbers = set()
    for struct in all_content_structures:
        all_section_numbers.update(struct['sections'].keys())
    
    # Sort section numbers numerically
    sorted_sections = sorted(all_section_numbers, key=int)
    
    print(f"Processing {len(sorted_sections)} main sections: {sorted_sections}")
    
    # Step 2: Process each main section
    for section_num in sorted_sections:
        print(f"\n--- Processing Section {section_num} ---")
        
        # Get the main heading (first heading) for this section from each PDF
        section_main_headings = []
        for struct in all_content_structures:
            if section_num in struct['sections']:
                section_headings = struct['sections'][section_num]
                if section_headings:
                    main_heading = section_headings[0]  # First heading in the section
                    section_main_headings.append({
                        'pdf_name': struct['pdf_name'],
                        'heading': main_heading['heading'],
                        'content': main_heading['content'],
                        'all_section_headings': section_headings
                    })
        
        if not section_main_headings:
            continue
        
        # Check if main headings are the same (using normalize)
        normalized_main_headings = [normalize_heading(h['heading']) for h in section_main_headings]
        sections_match = len(set(normalized_main_headings)) == 1
        
        if sections_match:
            # MAPPED: Sections match - combine all subsections together
            representative_main = section_main_headings[0]
            print(f"MAPPED: Section {section_num} - '{representative_main['heading']}' ({len(section_main_headings)} PDFs)")
            
            # Combine the main heading
            main_heading_content = {}
            sources = []
            for h in section_main_headings:
                main_heading_content[h['pdf_name']] = h['content']
                sources.append(h['pdf_name'])
            main_heading_content['sources'] = sources
            final_combined_ordered.append((representative_main['heading'], main_heading_content))
            
            # Collect all subsections from all PDFs for this section
            all_subsections = []
            for h in section_main_headings:
                # Add subsections (skip the first one as it's the main heading we already processed)
                subsections = h['all_section_headings'][1:]
                for subsection in subsections:
                    all_subsections.append({
                        'pdf_name': h['pdf_name'],
                        'heading': subsection['heading'],
                        'content': subsection['content']
                    })
            
            # Add all subsections in order
            for subsection in all_subsections:
                subsection_content = {
                    subsection['pdf_name']: subsection['content'],
                    'sources': [subsection['pdf_name']]
                }
                final_combined_ordered.append((subsection['heading'], subsection_content))
            
        else:
            # NOT MAPPED: Sections don't match - add all sections from each PDF separately
            print(f"NOT MAPPED: Section {section_num} - Different content across PDFs")
            
            for h in section_main_headings:
                print(f"  Adding all Section {section_num} headings from {h['pdf_name']}")
                # Add all headings from this section (main + subsections)
                for heading_data in h['all_section_headings']:
                    heading_content = {
                        h['pdf_name']: heading_data['content'],
                        'sources': [h['pdf_name']]
                    }
                    final_combined_ordered.append((heading_data['heading'], heading_content))
    
    # Step 3: Add other headings (non-numbered sections like "Appendix", etc.)
    print(f"\n--- Processing Other Headings ---")
    for struct in all_content_structures:
        if struct['other_headings']:
            print(f"Adding {len(struct['other_headings'])} other headings from {struct['pdf_name']}")
            for heading_data in struct['other_headings']:
                heading_content = {
                    struct['pdf_name']: heading_data['content'],
                    'sources': [struct['pdf_name']]
                }
                final_combined_ordered.append((heading_data['heading'], heading_content))
    
    # Convert ordered list to OrderedDict for JSON output to preserve order
    final_combined = OrderedDict()
    for heading, content in final_combined_ordered:
        final_combined[heading] = content
    
    # Save combined content
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_combined, f, ensure_ascii=False, indent=2)
    
    print(f"\nCombined text content saved to: {output_path}")
    print(f"Total headings: {len(final_combined)}")
    
    # Print summary
    mapped_count = sum(1 for v in final_combined.values() if len(v.get('sources', [])) > 1)
    non_mapped_count = len(final_combined) - mapped_count
    print(f"  - Mapped headings (same section): {mapped_count}")
    print(f"  - Non-mapped headings (different/separate): {non_mapped_count}")

# --- Utility Functions (kept for future use) ---
def normalize_paragraph(text):
    """Lowercase, strip, and remove leading bullets/specials from paragraph content."""
    text = text.lower().strip()
    # Replace unicode ellipsis with three dots
    text = text.replace('â€¦', '...')
    # Remove all leading punctuation, ellipsis, and whitespace
    text = re.sub(r'^[\s\.,;:Â·â€¢\-\*]+', '', text)
    # Remove repeated dots at the start (e.g. ... or ....)
    text = re.sub(r'^\.+', '', text)
    # Remove any remaining leading non-alphanumeric chars
    text = re.sub(r'^[^\w\d]+', '', text)
    return text

def calc_similarity(a, b):
    """Return SequenceMatcher similarity ratio between two strings."""
    return difflib.SequenceMatcher(None, a, b).ratio()

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
pdf_folder = r"C:\Users\prathyuc\GDS\GDS\final_pdfs"
output_folder = r"C:\Users\prathyuc\GDS\GDS\extracted_content"

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

def extract_tables_to_excel(all_tables, excel_path, paragraphs=None):
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

def process_pdf(pdf_name, pdf_folder, output_folder):
    """Process a single PDF file."""
    pdf_path = os.path.join(pdf_folder, pdf_name)
    if not os.path.isfile(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    print(f"Processing: {pdf_name}")
    
    # Get base filename without extension
    pdf_base = os.path.splitext(pdf_name)[0]
    
    # Create subfolder for this PDF
    pdf_output_folder = os.path.join(output_folder, pdf_base)
    os.makedirs(pdf_output_folder, exist_ok=True)
    
    # Analyze document with Document Intelligence
    with open(pdf_path, "rb") as f:
        poller = document_intelligence_client.begin_analyze_document(
            "prebuilt-layout", analyze_request=f, content_type="application/octet-stream"
        )
        result = poller.result()
    
    # Save only tables and paragraphs to results.json
    filtered_results = {
        "tables": [table.as_dict() for table in result.tables] if result.tables else [],
        "paragraphs": [paragraph.as_dict() for paragraph in result.paragraphs] if result.paragraphs else []
    }
    results_path = os.path.join(pdf_output_folder, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(filtered_results, f, ensure_ascii=False, indent=2)
    print(f"Saved tables and paragraphs to: {results_path}")
    
    # Extract layout data
    paragraphs = [p.as_dict() for p in result.paragraphs]
    all_tables = [t.as_dict() for t in result.tables] if hasattr(result, 'tables') else []
    
    # Combine split tables across pages
    combined_tables = combine_split_tables(all_tables)
    
    # Add internal index to each paragraph for element reference
    for i, para in enumerate(paragraphs):
        para['_internal_index'] = i
    
    # Build table paragraph indices using combined tables
    table_paragraph_indices = build_table_paragraph_index(combined_tables)
    
    # Extract content by headings using combined tables
    content_structure = extract_content_by_headings(paragraphs, combined_tables, table_paragraph_indices)
    
    # Save content to JSON
    json_path = os.path.join(pdf_output_folder, f"{pdf_base}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(content_structure, f, ensure_ascii=False, indent=2)
    print(f"Saved content to: {json_path}")
    
    # Save tables to Excel using combined tables
    if combined_tables:
        excel_path = os.path.join(pdf_output_folder, f"{pdf_base}.xlsx")
        extract_tables_to_excel(combined_tables, excel_path, paragraphs)
        print(f"Saved tables to: {excel_path}")
    else:
        print(f"No tables found in {pdf_name}")

def combine_all_pdf_text_content():
    """Combine text content from all processed PDFs in the output folder."""
    if not os.path.exists(output_folder):
        print(f"Output folder not found: {output_folder}")
        return
    
    # Find all PDF JSON files (pdf_name.json) but only from PDF-specific subfolders
    # Skip backup folders or other non-PDF folders
    json_files = []
    for item in os.listdir(output_folder):
        item_path = os.path.join(output_folder, item)
        # Only consider directories that are not backup folders
        if os.path.isdir(item_path) and not item.startswith('bckp') and not item.startswith('backup'):
            # Look for the PDF JSON file inside this directory
            pdf_json_file = os.path.join(item_path, f"{item}.json")
            if os.path.exists(pdf_json_file):
                json_files.append(pdf_json_file)
    
    if not json_files:
        print("No PDF JSON files found in output folder.")
        return
    
    print(f"Found {len(json_files)} text content files:")
    for json_file in json_files:
        print(f"  - {json_file}")
    
    # Create combined output path
    combined_output_path = os.path.join(output_folder, "combined_text_content.json")
    
    # Combine the content
    combine_text_content_from_pdfs(json_files, combined_output_path)

def main():
    """Main function to process all PDFs in the final_pdfs folder."""
    # Get all PDF files in the folder
    if not os.path.exists(pdf_folder):
        print(f"PDF folder not found: {pdf_folder}")
        return
    
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in: {pdf_folder}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process:")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file}")
    
    print("\nStarting processing...")
    
    # Process each PDF
    for pdf_file in pdf_files:
        try:
            process_pdf(pdf_file, pdf_folder, output_folder)
            print("-" * 50)
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
            print("-" * 50)
    
    print("Processing completed!")
    
    # Combine text content from all PDFs
    print("\n" + "=" * 50)
    print("COMBINING TEXT CONTENT FROM ALL PDFs")
    print("=" * 50)
    combine_all_pdf_text_content()

if __name__ == "__main__":
    main()

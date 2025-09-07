"""
Advanced PDF Extractor with Placement Tracking
Uses Azure Document Intelligence + PyMuPDF for comprehensive extraction with precise coordinates

Features:
- Azure DI: Superior table and text extraction with structure
- PyMuPDF: Image extraction and coordinate mapping
- Placement tracking: Exact coordinates for all elements
- Folder-based processing: input/ -> output/
"""

import os
import json
import re
import fitz  # PyMuPDF
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PlacementPDFExtractor:
    def __init__(self):
        """Initialize with Azure DI and configure logging"""
        self.setup_logging()
        self.setup_azure_client()
        self.input_folder = Path("input")
        self.output_folder = Path("output")
        self.ensure_folders()
    
    def setup_logging(self):
        """Configure logging for detailed tracking"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_azure_client(self):
        """Initialize Azure Document Intelligence client"""
        # Try both variable name formats
        endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT") or os.getenv("DI_ENDPOINT")
        key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY") or os.getenv("DI_KEY")
        
        if not endpoint or not key:
            self.logger.warning("Azure DI credentials not found in .env file")
            self.azure_client = None
        else:
            self.azure_client = DocumentIntelligenceClient(
                endpoint=endpoint,
                credential=AzureKeyCredential(key)
            )
            self.logger.info("Azure Document Intelligence client initialized")
    
    def ensure_folders(self):
        """Create input and output folders if they don't exist"""
        self.input_folder.mkdir(exist_ok=True)
        self.output_folder.mkdir(exist_ok=True)
        self.logger.info(f"Folders ready: {self.input_folder} -> {self.output_folder}")
    
    def process_all_pdfs(self):
        """Process all PDFs in the input folder"""
        pdf_files = list(self.input_folder.glob("*.pdf"))
        
        if not pdf_files:
            self.logger.warning(f"No PDF files found in {self.input_folder}")
            return
        
        self.logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            self.logger.info(f"Processing: {pdf_file.name}")
            try:
                self.extract_pdf_with_placement(pdf_file)
                self.logger.info(f"✅ Completed: {pdf_file.name}")
            except Exception as e:
                self.logger.error(f"❌ Failed {pdf_file.name}: {str(e)}")
    
    def extract_pdf_with_placement(self, pdf_path: Path):
        """Extract PDF with placement tracking using Azure DI + PyMuPDF"""
        pdf_name = pdf_path.stem
        output_dir = self.output_folder / pdf_name
        output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        images_dir = output_dir / "images"
        tables_dir = output_dir / "tables"
        
        # Clean up old table files to avoid confusion
        if tables_dir.exists():
            import shutil
            shutil.rmtree(tables_dir)
        
        images_dir.mkdir(exist_ok=True)
        tables_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Extracting to: {output_dir}")
        
        # Initialize data structures
        placement_data = {
            "document_info": {
                "filename": pdf_path.name,
                "total_pages": 0
            },
            "pages": {}
        }
        
        # Open PDF with PyMuPDF for images and coordinates
        doc = fitz.open(pdf_path)
        placement_data["document_info"]["total_pages"] = len(doc)
        
        # Extract images and hyperlinks with PyMuPDF
        self.extract_images_with_pymupdf(doc, images_dir, placement_data)
        self.extract_hyperlinks_with_pymupdf(doc, placement_data)
        
        # Extract text and tables with Azure DI (if available)
        if self.azure_client:
            self.extract_text_tables_with_azure(pdf_path, output_dir, placement_data)
        else:
            # Fallback to PyMuPDF for text and tables
            self.extract_text_tables_with_pymupdf(doc, output_dir, placement_data)
        
        doc.close()
        
        # Save placement data
        self.save_placement_data(output_dir, placement_data)
        self.create_summary_report(output_dir, placement_data)
    
    def is_template_image(self, image_data: dict, page_width: float, page_height: float) -> bool:
        """
        Determine if an image is a template/decorative element vs content image
        
        Template images are typically:
        - Very small (logos, icons)
        - Positioned at page edges (headers, footers)
        - Very large covering most of the page (backgrounds)
        - Have very low or very high DPI indicating graphics vs photos
        """
        
        width_points = image_data["width_points"]
        height_points = image_data["height_points"]
        bbox = image_data["bbox"]
        pixel_width = image_data["pixel_width"]
        pixel_height = image_data["pixel_height"]
        dpi_x = image_data["dpi_x"]
        
        # Calculate position ratios
        x_pos_ratio = bbox[0] / page_width if page_width > 0 else 0
        y_pos_ratio = bbox[1] / page_height if page_height > 0 else 0
        
        # Calculate size ratios
        width_ratio = width_points / page_width if page_width > 0 else 0
        height_ratio = height_points / page_height if page_height > 0 else 0
        area_ratio = (width_points * height_points) / (page_width * page_height) if (page_width * page_height) > 0 else 0
        
        # Rule 1: Very small images (likely logos, icons, bullets)
        if width_points < 50 or height_points < 50:
            return True
        
        # Rule 2: Very small pixel dimensions (likely vector graphics/icons)
        if pixel_width < 100 or pixel_height < 100:
            return True
        
        # Rule 3: Images in header area (top 10% of page)
        if y_pos_ratio < 0.1:
            return True
        
        # Rule 4: Images in footer area (bottom 10% of page)
        if y_pos_ratio > 0.9:
            return True
        
        # Rule 5: Images in left/right margins (first/last 5% of page width)
        if x_pos_ratio < 0.05 or x_pos_ratio > 0.95:
            return True
        
        # Rule 6: Very large images covering most of the page (backgrounds)
        if area_ratio > 0.8:
            return True
        
        # Rule 7: Very small area ratio (decorative elements)
        if area_ratio < 0.01:
            return True
        
        # Rule 8: Very high DPI images that are small (likely logos)
        if dpi_x > 300 and width_points < 100:
            return True
        
        # Rule 9: Very low DPI images (likely template graphics)
        if dpi_x < 72 and dpi_x > 0:
            return True
        
        # Rule 10: Square small images (likely icons or logos)
        aspect_ratio = width_points / height_points if height_points > 0 else 1
        if 0.8 <= aspect_ratio <= 1.2 and width_points < 150:
            return True
        
        # If none of the template rules match, it's likely a content image
        return False

    def extract_images_with_pymupdf(self, doc, images_dir: Path, placement_data: Dict):
        """Enhanced image extraction with template filtering - only content images saved"""
        self.logger.info("Extracting content images (filtering out template images)...")
        
        total_images = 0
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_key = f"page_{page_num + 1}"
            
            # Enhanced page data with transformation matrix
            page_matrix = page.transformation_matrix
            page_rotation = page.rotation
            
            # Initialize page data
            if page_key not in placement_data["pages"]:
                placement_data["pages"][page_key] = {
                    "page_size": {"width": page.rect.width, "height": page.rect.height},
                    "page_rotation": page_rotation,
                    "transformation_matrix": list(page_matrix),
                    "images": [],
                    "tables": [],
                    "text_blocks": [],
                    "hyperlinks": []
                }
            
            # Get images with full metadata
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                try:
                    # Get image data with enhanced metadata
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Get image rectangles (placement) with precise coordinates
                    img_rects = page.get_image_rects(img)
                    
                    for rect_index, rect in enumerate(img_rects):
                        # Enhanced filename with metadata
                        img_filename = f"page_{page_num + 1}_img_{img_index + 1}_{rect_index + 1}.png"
                        
                        # Calculate enhanced placement metrics first
                        width_points = rect.width
                        height_points = rect.height
                        pixel_width = base_image.get("width", 0)
                        pixel_height = base_image.get("height", 0)
                        
                        # Calculate DPI (dots per inch)
                        dpi_x = pixel_width / (width_points / 72) if width_points > 0 else 0
                        dpi_y = pixel_height / (height_points / 72) if height_points > 0 else 0
                        
                        # Enhanced placement data with spatial intelligence
                        image_data = {
                            "id": f"img_{total_images + 1}",
                            "filename": img_filename,
                            "bbox": [rect.x0, rect.y0, rect.x1, rect.y1],
                            "width_points": width_points,
                            "height_points": height_points,
                            "pixel_width": pixel_width,
                            "pixel_height": pixel_height,
                            "dpi_x": round(dpi_x, 2),
                            "dpi_y": round(dpi_y, 2),
                            "page": page_num + 1,
                            "format": base_image.get("ext", "png"),
                            "xref": xref,
                            "z_order": img_index,  # Layer order on page
                            "colorspace": base_image.get("colorspace", "unknown"),
                            "bits_per_component": base_image.get("bpc", 8),
                            "center_x": rect.x0 + width_points / 2,
                            "center_y": rect.y0 + height_points / 2,
                            "area_points": width_points * height_points
                        }
                        
                        # Check if this is a template image (to be filtered out)
                        page_width = placement_data["pages"][page_key]["page_size"]["width"]
                        page_height = placement_data["pages"][page_key]["page_size"]["height"]
                        
                        if self.is_template_image(image_data, page_width, page_height):
                            self.logger.debug(f"Skipping template image: {img_filename} (size: {width_points:.0f}x{height_points:.0f}, pos: {rect.x0:.0f},{rect.y0:.0f})")
                            continue  # Skip template images
                        
                        # Only save and record content images
                        img_path = images_dir / img_filename
                        with open(img_path, "wb") as img_file:
                            img_file.write(image_bytes)
                        
                        # Mark as content image
                        image_data["image_type"] = "content"
                        
                        placement_data["pages"][page_key]["images"].append(image_data)
                        total_images += 1
                        
                        self.logger.debug(f"Saved content image: {img_filename} (size: {width_points:.0f}x{height_points:.0f})")
                
                except Exception as e:
                    self.logger.warning(f"Failed to extract image {img_index} from page {page_num + 1}: {e}")
        
        self.logger.info(f"Extracted {total_images} content images (template images filtered out)")
    
    def extract_hyperlinks_with_pymupdf(self, doc, placement_data: Dict):
        """Extract hyperlinks with coordinates using PyMuPDF"""
        self.logger.info("Extracting hyperlinks with placement...")
        
        total_links = 0
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_key = f"page_{page_num + 1}"
            
            links = page.get_links()
            
            for link in links:
                if link.get("uri"):  # External link
                    link_data = {
                        "id": f"link_{total_links + 1}",
                        "url": link["uri"],
                        "bbox": [link["from"].x0, link["from"].y0, link["from"].x1, link["from"].y1],
                        "page": page_num + 1,
                        "type": "external"
                    }
                    
                    placement_data["pages"][page_key]["hyperlinks"].append(link_data)
                    total_links += 1
                
                elif link.get("page") is not None:  # Internal link
                    link_data = {
                        "id": f"link_{total_links + 1}",
                        "target_page": link["page"] + 1,
                        "bbox": [link["from"].x0, link["from"].y0, link["from"].x1, link["from"].y1],
                        "page": page_num + 1,
                        "type": "internal"
                    }
                    
                    placement_data["pages"][page_key]["hyperlinks"].append(link_data)
                    total_links += 1
        
        self.logger.info(f"Extracted {total_links} hyperlinks with placement data")
    
    def get_table_headers(self, table) -> List[str]:
        """Extract column headers from a table."""
        headers = []
        # Find all cells in row 0 (header row)
        header_cells = [cell for cell in table.cells if cell.row_index == 0]
        # Sort by column index
        header_cells.sort(key=lambda x: x.column_index)
        
        for cell in header_cells:
            headers.append(cell.content.strip() if cell.content else "")
        
        return headers

    def get_table_page_number(self, table) -> int:
        """Get the page number of a table."""
        if hasattr(table, 'bounding_regions') and table.bounding_regions:
            return table.bounding_regions[0].page_number
        return 0

    def are_headers_similar(self, headers1: List[str], headers2: List[str], similarity_threshold: float = 0.8) -> bool:
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
        
        similarity = matches / len(headers1) if len(headers1) > 0 else 0
        return similarity >= similarity_threshold

    def combine_table_parts(self, table_parts: List) -> 'AzureTable':
        """Combine multiple table parts into a single table."""
        if len(table_parts) == 1:
            return table_parts[0]
        
        # Use the first table as the base
        combined_table = table_parts[0]
        combined_cells = list(combined_table.cells)
        
        # Find the maximum row index in the first table
        max_row_idx = max(cell.row_index for cell in combined_cells) if combined_cells else -1
        
        # Add cells from subsequent tables, adjusting row indices
        for table_part in table_parts[1:]:
            # Skip header row (row 0) in continuation tables
            non_header_cells = [cell for cell in table_part.cells if cell.row_index > 0]
            
            for cell in non_header_cells:
                # Create a copy of cell data and adjust row index
                # Note: We can't modify the original cell object, so we'll work with the combined table's cell list
                # This is a simplified approach - Azure DI objects are complex
                new_cell_data = {
                    'row_index': cell.row_index + max_row_idx,
                    'column_index': cell.column_index,
                    'row_span': getattr(cell, 'row_span', 1),
                    'column_span': getattr(cell, 'column_span', 1),
                    'content': cell.content,
                    'confidence': getattr(cell, 'confidence', 1.0)
                }
                
                # Create a mock cell object (simplified)
                class MockCell:
                    def __init__(self, data):
                        self.row_index = data['row_index']
                        self.column_index = data['column_index']
                        self.row_span = data['row_span']
                        self.column_span = data['column_span']
                        self.content = data['content']
                        self.confidence = data['confidence']
                
                combined_cells.append(MockCell(new_cell_data))
            
            # Update max_row_idx for the next iteration
            if non_header_cells:
                max_row_idx = max(cell.row_index + max_row_idx for cell in non_header_cells if cell.row_index > 0)
        
        # Replace the cells in the combined table
        combined_table.cells = combined_cells
        
        return combined_table

    def extract_table_header_and_context(self, page_text: str, table_bbox: List[float], page_num: int) -> Dict[str, str]:
        """Extract table header/caption and surrounding context for placement using intelligent section detection."""
        if not table_bbox or len(table_bbox) < 4:
            return {"header": "", "context": "", "placement_key": f"page_{page_num}_table"}
        
        lines = page_text.split('\n')
        
        table_header = ""
        context_lines = []
        last_section_heading = ""
        
        # Step 1: Look for explicit table captions/references
        explicit_table_patterns = [
            r'^Table\s+[\d\-A-Z]+[:\.]?\s*(.+)',     # "Table A-1: RACI Chart"
            r'^Figure\s+[\d\-A-Z]+[:\.]?\s*(.+)',    # "Figure 1-1: Process"
            r'^([A-Z]\.\d+\s+RACI\s+Chart[^-]*)',    # "A.1 RACI Chart - Basic"
        ]
        
        # Step 2: Look for section headings (more comprehensive patterns)
        section_patterns = [
            r'^(\d+\s+[A-Z][a-z]+.*)',               # "3 Responsibilities", "5 Design Requirements"
            r'^(\d+\.\d+\s+[A-Z][a-z]+.*)',          # "4.1 General", "4.1.1 Modified Design"
            r'^(\d+\.\d+\.\d+\s+[A-Z][a-z]+.*)',     # "4.1.1.1 TIMP Piping"
            r'^([A-Z]\s+[A-Z][a-z]+.*)',             # "A Purpose", "B Scope"
            r'^(Appendix\s+[A-Z][:\s]+.*)',          # "Appendix A: RACI Charts"
        ]
        
        # Find explicit table captions first (highest priority)
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            for pattern in explicit_table_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    caption = match.group(1) if match.groups() else line
                    if len(caption.strip()) < 100:  # Reasonable length for table caption
                        table_header = caption.strip()
                        break
            
            if table_header:
                break
        
        # If no explicit caption found, track the most recent section heading
        if not table_header:
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if this line is a section heading
                for pattern in section_patterns:
                    match = re.match(pattern, line, re.IGNORECASE)
                    if match:
                        section = match.group(1) if match.groups() else line
                        if len(section.strip()) < 80:  # Reasonable length for section heading
                            last_section_heading = section.strip()
                        break
        
        # Use the most recent section heading as table header if no explicit caption
        if not table_header and last_section_heading:
            table_header = last_section_heading
        
        # Final fallback
        if not table_header:
            table_header = f"Table on Page {page_num}"
        
        # Create meaningful context
        context_keywords = ['responsibilities', 'roles', 'accountabilities', 'raci', 'chart', 'table']
        for line in lines:
            line_clean = line.strip()
            if (line_clean and len(line_clean) < 150 and 
                any(keyword in line_clean.lower() for keyword in context_keywords) and
                line_clean != table_header):
                context_lines.append(line_clean)
                if len(context_lines) >= 3:  # Limit context to avoid noise
                    break
        
        context = " | ".join(context_lines)
        
        # Create placement key
        placement_key = table_header.lower()
        placement_key = re.sub(r'[^\w\s]', '', placement_key)
        placement_key = re.sub(r'\s+', '_', placement_key.strip())
        
        return {
            "header": table_header,
            "context": context,
            "placement_key": placement_key,
            "page": page_num
        }

    def combine_split_tables(self, tables: List) -> List:
        """Combine tables that are split across multiple pages."""
        if not tables:
            return tables
        
        self.logger.info(f"Processing {len(tables)} tables for potential combination...")
        
        # Sort tables by page number
        tables_with_pages = []
        for i, table in enumerate(tables):
            page_num = self.get_table_page_number(table)
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
            current_headers = self.get_table_headers(current_table)
            
            # Look for tables on subsequent pages with similar headers
            tables_to_combine = [current_table]
            last_page = current_page
            
            j = i + 1
            while j < len(tables_with_pages):
                next_page, next_idx, next_table = tables_with_pages[j]
                next_headers = self.get_table_headers(next_table)
                
                # Check if it's on the next page and has similar headers
                if (next_page == last_page + 1 and 
                    self.are_headers_similar(current_headers, next_headers)):
                    
                    tables_to_combine.append(next_table)
                    skip_indices.add(j)
                    last_page = next_page
                    self.logger.info(f"Found continuation: Table on page {current_page} continues on page {next_page}")
                    j += 1
                else:
                    break
            
            # Combine the tables if we found continuations
            if len(tables_to_combine) > 1:
                combined_table = self.combine_table_parts(tables_to_combine)
                combined_tables.append(combined_table)
                self.logger.info(f"Combined {len(tables_to_combine)} table parts from pages {current_page} to {last_page}")
            else:
                combined_tables.append(current_table)
            
            i += 1
        
        self.logger.info(f"Resulted in {len(combined_tables)} tables after combination.")
        return combined_tables
    
    def get_pymupdf_table_headers(self, table_data: List[List[str]]) -> List[str]:
        """Extract headers from PyMuPDF table data."""
        if not table_data or not table_data[0]:
            return []
        return [str(cell).strip() for cell in table_data[0]]
    
    def are_pymupdf_headers_similar(self, headers1: List[str], headers2: List[str], similarity_threshold: float = 0.8) -> bool:
        """Check if two PyMuPDF header lists are similar."""
        return self.are_headers_similar(headers1, headers2, similarity_threshold)
    
    def combine_pymupdf_table_data(self, table_parts: List[List[List[str]]]) -> List[List[str]]:
        """Combine multiple PyMuPDF table data parts into a single table."""
        if len(table_parts) == 1:
            return table_parts[0]
        
        # Start with the first table (including headers)
        combined_data = table_parts[0].copy()
        
        # Add data from subsequent tables (skip header row)
        for table_part in table_parts[1:]:
            if len(table_part) > 1:  # Skip if only header row
                combined_data.extend(table_part[1:])  # Skip first row (header)
        
        return combined_data
    
    def _extract_table_header(self, page, table_bbox, page_num):
        """Extract table header/caption from text above or near the table with improved precision"""
        table_x0, table_y0, table_x1, table_y1 = table_bbox
        
        # First, look for text blocks directly above the table (traditional approach)
        search_area = {
            'x0': table_x0 - 30,  # Narrower search area 
            'y0': table_y0 - 80,  # Look 80 points above (reduced from 100)
            'x1': table_x1 + 30,  # Narrower search area
            'y1': table_y0 + 5    # Just slightly into table area
        }
        
        header_candidates = []
        section_candidates = []
        
        # Get text blocks in the search area
        text_blocks = page.get_text("dict")["blocks"]
        
        # First pass: Look for immediate table headers
        for block in text_blocks:
            if "lines" in block:
                for line in block["lines"]:
                    line_bbox = line["bbox"]
                    
                    # Check if line is in search area
                    if (line_bbox[0] >= search_area['x0'] and 
                        line_bbox[1] >= search_area['y0'] and
                        line_bbox[2] <= search_area['x1'] and 
                        line_bbox[3] <= search_area['y1']):
                        
                        line_text = ""
                        font_size = 0
                        is_bold = False
                        
                        for span in line["spans"]:
                            line_text += span["text"]
                            font_size = max(font_size, span.get("size", 0))
                            if span.get("flags", 0) & 2**4:  # Bold flag
                                is_bold = True
                        
                        line_text = line_text.strip()
                        if line_text and len(line_text) > 3:  # Skip very short text
                            # Calculate distance from table
                            distance = table_y0 - line_bbox[3]
                            
                            # Calculate header likelihood score
                            score = self._calculate_table_header_score(line_text, is_bold, font_size, distance)
                            
                            header_candidates.append({
                                'text': line_text,
                                'distance': distance,
                                'bbox': line_bbox,
                                'font_size': font_size,
                                'is_bold': is_bold,
                                'score': score
                            })
        
        # Second pass: Look for section headings throughout the page (for tables without explicit captions)
        for block in text_blocks:
            if "lines" in block:
                for line in block["lines"]:
                    line_bbox = line["bbox"]
                    
                    # Only consider text that is above the table (not below)
                    if line_bbox[3] < table_y0:
                        line_text = ""
                        font_size = 0
                        is_bold = False
                        
                        for span in line["spans"]:
                            line_text += span["text"]
                            font_size = max(font_size, span.get("size", 0))
                            if span.get("flags", 0) & 2**4:  # Bold flag
                                is_bold = True
                        
                        line_text = line_text.strip()
                        
                        # Check if this looks like a section heading
                        if self._is_section_heading(line_text, is_bold, font_size):
                            distance = table_y0 - line_bbox[3]
                            section_candidates.append({
                                'text': line_text,
                                'distance': distance,
                                'bbox': line_bbox,
                                'font_size': font_size,
                                'is_bold': is_bold,
                                'score': 70 if is_bold else 60  # Section headings get moderate score
                            })
        
        # Sort candidates by score first (highest), then by distance (closest)
        header_candidates.sort(key=lambda x: (-x['score'], x['distance']))
        section_candidates.sort(key=lambda x: (-x['distance']))  # For sections, use the closest (most recent)
        
        # Select best header
        best_header = None
        context_lines = []
        
        # First, check for high-confidence explicit table headers
        for candidate in header_candidates[:5]:  # Check top 5 candidates
            if candidate['score'] >= 50:  # High confidence threshold
                best_header = candidate
                break
            elif candidate['distance'] < 40:  # Close to table
                context_lines.append(candidate['text'])
        
        # If no explicit header found, use the most recent section heading
        if not best_header and section_candidates:
            best_header = section_candidates[0]  # Most recent section heading
        
        # Build result
        if best_header:
            header_text = best_header['text']
            context_text = ' | '.join([c['text'] for c in header_candidates[:3] if c != best_header])
        elif context_lines:
            # Use first line of nearby context as header
            header_text = context_lines[0]
            context_text = ' | '.join(context_lines)
        else:
            # Last resort: use generic identifier
            header_text = f"Table {page_num}-{int(table_y0)}"  # Page + Y position
            context_text = ""
        
        return {
            'header': header_text,
            'context': context_text,
            'candidates': header_candidates,
            'table_bbox': table_bbox
        }
    
    def _is_section_heading(self, text: str, is_bold: bool, font_size: float) -> bool:
        """Check if text looks like a section heading"""
        text = text.strip()
        
        # Pattern matching for section headings
        section_patterns = [
            r'^\d+\s+[A-Z][a-z]+.*',               # "3 Responsibilities", "5 Design Requirements"
            r'^\d+\.\d+\s+[A-Z][a-z]+.*',          # "4.1 General", "4.1.1 Modified Design"
            r'^\d+\.\d+\.\d+\s+[A-Z][a-z]+.*',     # "4.1.1.1 TIMP Piping"
            r'^[A-Z]\.\d+\s+[A-Z]+.*',             # "A.1 RACI Chart"
            r'^Appendix\s+[A-Z][:\s]+.*',          # "Appendix A: RACI Charts"
        ]
        
        # Check if text matches section heading patterns
        for pattern in section_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        
        # Additional checks for bold text that could be headings
        if is_bold and font_size >= 12 and len(text) < 80:
            # Check for keyword-based headings
            heading_keywords = ['purpose', 'scope', 'responsibilities', 'requirements', 'design', 'appendix']
            if any(keyword in text.lower() for keyword in heading_keywords):
                return True
        
        return False

    def _calculate_table_header_score(self, text: str, is_bold: bool, font_size: float, distance: float) -> float:
        """Calculate likelihood score for text being a table header"""
        score = 0.0
        text_lower = text.lower().strip()
        
        # Strong positive indicators
        if re.match(r'^table\s+[a-z]?[\d\-\.]+', text_lower):  # "Table A-1", "Table 4-1"
            score += 100.0
        elif re.match(r'^[a-z]\.\d+\s+\w+\s+(chart|table|matrix)', text_lower):  # "A.1 RACI Chart"
            score += 95.0
        elif re.match(r'^\w+\s+(chart|matrix|schedule|table)$', text_lower):  # "RACI Chart", "Design Matrix"
            score += 80.0
        elif 'table' in text_lower:
            score += 40.0
        elif any(word in text_lower for word in ['chart', 'matrix', 'schedule', 'appendix']):
            score += 30.0
        
        # Formatting bonuses
        if is_bold:
            score += 20.0
        if font_size > 12:
            score += 10.0
        
        # Length penalties/bonuses
        word_count = len(text.split())
        if 2 <= word_count <= 10:  # Good title length
            score += 10.0
        elif word_count > 20:  # Too long for a title
            score -= 30.0
        elif word_count == 1:  # Single words are rarely good headers
            score -= 10.0
        
        # Distance penalty (farther = less likely to be header)
        if distance > 50:
            score -= 20.0
        elif distance < 10:
            score += 10.0
        
        # Penalty for paragraph-like text
        if len(text) > 100 and '.' in text:
            score -= 25.0
        
        # Penalty for clearly non-header text
        if any(phrase in text_lower for phrase in ['the following', 'as described', 'this document', 'note:']):
            score -= 40.0
        
        return max(score, 0.0)
    
    def _create_placement_key(self, header: str, page_num: int) -> str:
        """Create a placement key from table header"""
        if not header:
            return f"table_page_{page_num}"
        
        # Convert to lowercase and clean
        key = header.lower()
        key = re.sub(r'[^\w\s\-\.]', '', key)  # Keep only word chars, spaces, hyphens, dots
        key = re.sub(r'\s+', '_', key.strip())  # Replace spaces with underscores
        key = re.sub(r'_{2,}', '_', key)  # Remove multiple underscores
        
        return key
    
    def combine_split_pymupdf_tables(self, table_info_list: List[Dict]) -> List[Dict]:
        """Combine PyMuPDF tables that are split across pages."""
        if not table_info_list:
            return table_info_list
        
        self.logger.info(f"Processing {len(table_info_list)} PyMuPDF tables for potential combination...")
        
        # Sort by page number
        sorted_tables = sorted(table_info_list, key=lambda x: x['page'])
        
        combined_tables = []
        skip_indices = set()
        
        i = 0
        while i < len(sorted_tables):
            if i in skip_indices:
                i += 1
                continue
                
            current_table = sorted_tables[i]
            current_headers = self.get_pymupdf_table_headers(current_table['data'])
            
            # Look for tables on subsequent pages with similar headers
            tables_to_combine = [current_table]
            last_page = current_table['page']
            
            j = i + 1
            while j < len(sorted_tables):
                next_table = sorted_tables[j]
                next_headers = self.get_pymupdf_table_headers(next_table['data'])
                
                # Check if it's on the next page and has similar headers
                if (next_table['page'] == last_page + 1 and 
                    self.are_pymupdf_headers_similar(current_headers, next_headers)):
                    
                    tables_to_combine.append(next_table)
                    skip_indices.add(j)
                    last_page = next_table['page']
                    self.logger.info(f"Found PyMuPDF continuation: Table on page {current_table['page']} continues on page {next_table['page']}")
                    j += 1
                else:
                    break
            
            # Combine the tables if we found continuations
            if len(tables_to_combine) > 1:
                # Combine table data
                combined_data = self.combine_pymupdf_table_data([t['data'] for t in tables_to_combine])
                
                # Preserve table header from the first table (where the header appears)
                first_table = tables_to_combine[0]
                table_header = first_table.get('table_header', {})
                
                # Create combined table info
                combined_table = {
                    'data': combined_data,
                    'page': first_table['page'],  # Start page
                    'bbox': first_table['bbox'],  # Use first table's bbox
                    'table_header': table_header,  # Preserve header from first part
                    'content_context': table_header.get('context', ''),
                    'combined_from_pages': [t['page'] for t in tables_to_combine],
                    'is_combined': True
                }
                
                combined_tables.append(combined_table)
                self.logger.info(f"Combined {len(tables_to_combine)} PyMuPDF table parts from pages {first_table['page']} to {last_page}")
            else:
                # Single table - ensure it has the header info
                single_table = current_table
                if 'table_header' not in single_table:
                    single_table['table_header'] = {'header': '', 'context': ''}
                if 'content_context' not in single_table:
                    single_table['content_context'] = single_table['table_header'].get('context', '')
                single_table['is_combined'] = False
                
                combined_tables.append(single_table)
            
            i += 1
        
        self.logger.info(f"Resulted in {len(combined_tables)} PyMuPDF tables after combination.")
        return combined_tables
    
    def extract_text_tables_with_azure(self, pdf_path: Path, output_dir: Path, placement_data: Dict):
        """Extract text and tables using Azure Document Intelligence"""
        self.logger.info("Extracting text and tables with Azure Document Intelligence...")
        
        try:
            # Read PDF file
            with open(pdf_path, "rb") as pdf_file:
                pdf_bytes = pdf_file.read()
            
            # Analyze document
            poller = self.azure_client.begin_analyze_document(
                "prebuilt-layout",
                pdf_bytes,
                content_type="application/pdf"
            )
            
            result = poller.result()
            
            # Extract text with placement
            self.process_azure_text(result, output_dir, placement_data)
            
            # Extract tables with placement
            self.process_azure_tables(result, output_dir, placement_data)
            
        except Exception as e:
            self.logger.error(f"Azure DI extraction failed: {e}")
            self.logger.info("Falling back to PyMuPDF...")
            # Fallback to PyMuPDF
            doc = fitz.open(pdf_path)
            self.extract_text_tables_with_pymupdf(doc, output_dir, placement_data)
            doc.close()
    
    def process_azure_text(self, result, output_dir: Path, placement_data: Dict):
        """Process text from Azure DI with smart filtering for headers, footers, and table content"""
        all_text = []
        
        # First pass: Extract all tables and their regions to exclude from text
        table_regions = {}
        for table in result.tables:
            if hasattr(table, 'bounding_regions') and table.bounding_regions:
                page_num = table.bounding_regions[0].page_number
                if page_num not in table_regions:
                    table_regions[page_num] = []
                
                polygon = table.bounding_regions[0].polygon
                if len(polygon) >= 4:
                    # Calculate table bounding box
                    x_coords = [point.x for point in polygon]
                    y_coords = [point.y for point in polygon]
                    table_bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                    table_regions[page_num].append(table_bbox)
        
        for page_num, page in enumerate(result.pages):
            page_key = f"page_{page_num + 1}"
            page_text = f"\n=== PAGE {page_num + 1} ===\n"
            
            # Get page dimensions for zone calculations
            page_width = page.width if hasattr(page, 'width') else 612
            page_height = page.height if hasattr(page, 'height') else 792
            
            # Define header, footer zones (more precise)
            header_zone_bottom = page_height * 0.12  # Top 12% is header
            footer_zone_top = page_height * 0.88     # Bottom 12% is footer
            
            # Process text lines with intelligent filtering
            filtered_lines = []
            hyperlink_areas = []
            
            # Get hyperlink areas for this page
            if page_key in placement_data["pages"]:
                hyperlink_areas = [link["bbox"] for link in placement_data["pages"][page_key].get("hyperlinks", [])]
            
            for line in page.lines:
                if not line.content or not line.content.strip():
                    continue
                
                # Get line coordinates
                line_bbox = [
                    line.polygon[0].x, line.polygon[0].y,
                    line.polygon[2].x, line.polygon[2].y
                ] if line.polygon and len(line.polygon) >= 4 else [0, 0, 0, 0]
                
                line_y = line_bbox[1]  # Top Y coordinate
                line_content = line.content.strip()
                
                # Skip if in header zone and looks like header content
                if line_y < header_zone_bottom and self.is_header_content(line_content):
                    self.logger.debug(f"Skipping header content: {line_content[:50]}")
                    continue
                
                # Skip if in footer zone and looks like footer content  
                if line_y > footer_zone_top and self.is_footer_content(line_content):
                    self.logger.debug(f"Skipping footer content: {line_content[:50]}")
                    continue
                
                # Skip if line is inside any table region
                if self.is_line_in_table_region(line_bbox, table_regions.get(page_num + 1, [])):
                    self.logger.debug(f"Skipping table content: {line_content[:50]}")
                    continue
                
                # Check if line contains hyperlink and mark it
                line_text = line_content
                if self.line_contains_hyperlink(line_bbox, hyperlink_areas):
                    line_text = f"🔗 {line_content} [HYPERLINK]"
                
                filtered_lines.append(line_text)
                
                # Store text block with placement (for coordinate tracking)
                text_block = {
                    "content": line_content,
                    "bbox": line_bbox,
                    "page": page_num + 1,
                    "confidence": getattr(line, 'confidence', 1.0),
                    "is_hyperlink": self.line_contains_hyperlink(line_bbox, hyperlink_areas),
                    "zone": self.get_text_zone(line_y, page_height)
                }
                
                if page_key in placement_data["pages"]:
                    placement_data["pages"][page_key]["text_blocks"].append(text_block)
            
            # Add filtered content to page text
            if filtered_lines:
                page_text += "\n".join(filtered_lines) + "\n"
            else:
                page_text += "[No main content text on this page]\n"
            
            all_text.append(page_text)
        
        # Save complete filtered text
        text_file = output_dir / "text_with_pages.txt"
        with open(text_file, "w", encoding="utf-8") as f:
            f.write("\n".join(all_text))
        
        # Save clean content only (no page markers)
        clean_text = []
        for page_text in all_text:
            lines = page_text.split('\n')
            content_lines = [line for line in lines if not line.startswith('=== PAGE')]
            if content_lines:
                clean_text.extend([line for line in content_lines if line.strip()])
        
        clean_file = output_dir / "clean_content_text.txt"
        with open(clean_file, "w", encoding="utf-8") as f:
            f.write("\n".join(clean_text))
        
        self.logger.info(f"Extracted and filtered text from {len(result.pages)} pages")
        self.logger.info(f"Saved clean content to: {clean_file}")
    
    def is_header_content(self, content: str) -> bool:
        """Identify if content is likely header material"""
        content_lower = content.lower().strip()
        
        # Common header patterns
        header_patterns = [
            "ploughing standard",
            "document control",
            "revision history", 
            "document governance",
            "page header",
            "title:",
            "subject:",
            "document id:",
            "version:",
            "date:",
            "prepared by:",
            "approved by:"
        ]
        
        # Check for header patterns
        for pattern in header_patterns:
            if pattern in content_lower:
                return True
        
        # Very short single words at top (likely titles)
        if len(content.split()) <= 3 and len(content) < 50:
            return True
        
        return False
    
    def is_footer_content(self, content: str) -> bool:
        """Identify if content is likely footer material"""
        content_lower = content.lower().strip()
        
        # Common footer patterns
        footer_patterns = [
            "publication date:",
            "page ",
            " of ",
            "effective:",
            "uncontrolled when printed",
            "controlled copy is in",
            "© 202",  # Copyright
            "enbridge inc",
            "gds document library",
            "st-",  # Document IDs
            "revision",
            "confidential"
        ]
        
        # Check for footer patterns
        for pattern in footer_patterns:
            if pattern in content_lower:
                return True
        
        # Page number patterns
        import re
        if re.search(r'\bpage\s+\d+\s+of\s+\d+\b', content_lower):
            return True
        
        # Document ID patterns (ST-XXXX-XXXX format)
        if re.search(r'st-[0-9a-f]{2,4}-[0-9a-f]{4}', content_lower):
            return True
        
        return False
    
    def is_line_in_table_region(self, line_bbox: list, table_regions: list) -> bool:
        """Check if a text line is inside any table region"""
        if not table_regions or not line_bbox:
            return False
        
        line_x1, line_y1, line_x2, line_y2 = line_bbox
        line_center_x = (line_x1 + line_x2) / 2
        line_center_y = (line_y1 + line_y2) / 2
        
        for table_bbox in table_regions:
            table_x1, table_y1, table_x2, table_y2 = table_bbox
            
            # Check if line center is inside table bounds (with small margin)
            margin = 5  # 5 point margin
            if (table_x1 - margin <= line_center_x <= table_x2 + margin and
                table_y1 - margin <= line_center_y <= table_y2 + margin):
                return True
        
        return False
    
    def line_contains_hyperlink(self, line_bbox: list, hyperlink_areas: list) -> bool:
        """Check if a text line overlaps with any hyperlink area"""
        if not hyperlink_areas or not line_bbox:
            return False
        
        line_x1, line_y1, line_x2, line_y2 = line_bbox
        
        for link_bbox in hyperlink_areas:
            link_x1, link_y1, link_x2, link_y2 = link_bbox
            
            # Check for overlap between line and hyperlink
            if not (line_x2 < link_x1 or line_x1 > link_x2 or 
                   line_y2 < link_y1 or line_y1 > link_y2):
                return True
        
        return False
    
    def get_text_zone(self, y_position: float, page_height: float) -> str:
        """Determine which zone of the page the text is in"""
        y_ratio = y_position / page_height
        
        if y_ratio < 0.12:
            return "header"
        elif y_ratio > 0.88:
            return "footer"
        else:
            return "body"
    
    def process_azure_tables(self, result, output_dir: Path, placement_data: Dict):
        """Process tables from Azure DI with placement and split table combination"""
        tables_dir = output_dir / "tables"
        
        # First, combine split tables across pages
        combined_tables = self.combine_split_tables(result.tables)
        
        tables_data = []
        
        for table_num, table in enumerate(combined_tables):
            # Extract table data
            table_data = []
            max_row = max([cell.row_index for cell in table.cells]) + 1
            max_col = max([cell.column_index for cell in table.cells]) + 1
            
            # Initialize table grid
            table_grid = [["" for _ in range(max_col)] for _ in range(max_row)]
            
            # Fill the grid
            for cell in table.cells:
                table_grid[cell.row_index][cell.column_index] = cell.content or ""
            
            # Convert to DataFrame and save
            df = pd.DataFrame(table_grid)
            
            # Get page number from the first bounding region
            page_num = self.get_table_page_number(table)
            excel_file = tables_dir / f"table_{table_num + 1}_page_{page_num}.xlsx"
            df.to_excel(excel_file, index=False, header=False)
            
            self.logger.info(f"Saved combined table: {excel_file}")
            
            # Store table placement data
            table_placement = {
                "id": f"table_{table_num + 1}",
                "filename": f"table_{table_num + 1}_page_{page_num}.xlsx",
                "rows": max_row,
                "columns": max_col,
                "bbox": [
                    table.bounding_regions[0].polygon[0].x,
                    table.bounding_regions[0].polygon[0].y,
                    table.bounding_regions[0].polygon[2].x,
                    table.bounding_regions[0].polygon[2].y
                ] if table.bounding_regions and table.bounding_regions[0].polygon else [0, 0, 0, 0],
                "page": page_num,
                "confidence": getattr(table, 'confidence', 1.0),
                "combined_from_pages": len(table.bounding_regions) if hasattr(table, 'bounding_regions') else 1
            }
            
            # Add to placement data
            page_key = f"page_{page_num}"
            if page_key in placement_data["pages"]:
                placement_data["pages"][page_key]["tables"].append(table_placement)
            
            tables_data.append(table_placement)
        
        # Save table locations
        tables_locations_file = tables_dir / "table_locations.json"
        with open(tables_locations_file, "w", encoding="utf-8") as f:
            json.dump(tables_data, f, indent=2)
        
        self.logger.info(f"Processed {len(combined_tables)} tables (after combining split tables)")
        
        return len(combined_tables)
        
        self.logger.info(f"Extracted {len(result.tables)} tables with placement data")
    
    def consolidate_text_blocks(self, page_data: Dict) -> List[Dict]:
        """Consolidate fragmented text spans into coherent paragraphs"""
        text_blocks = page_data.get("text_blocks", [])
        if not text_blocks:
            return []
        
        # Sort by reading order for proper consolidation
        text_blocks.sort(key=lambda x: x.get("reading_order", 0))
        
        consolidated_blocks = []
        current_block = None
        
        for block in text_blocks:
            content = block.get("content", "").strip()
            
            # Skip empty or whitespace-only content
            if not content:
                continue
            
            # Start new consolidated block or continue existing one
            if current_block is None:
                # Start new block
                current_block = {
                    "content": content,
                    "bbox": block["bbox"][:],  # Copy bbox
                    "page": block["page"],
                    "font": block.get("font", ""),
                    "font_size": block.get("font_size", 0),
                    "font_flags": block.get("font_flags", 0),
                    "color": block.get("color", 0),
                    "is_bold": block.get("is_bold", False),
                    "is_italic": block.get("is_italic", False),
                    "block_id": block.get("block_id", 0),
                    "spans": [block],  # Keep track of original spans
                    "reading_order": block.get("reading_order", 0)
                }
            else:
                # Check if this block should be merged with current
                should_merge = self._should_merge_text_blocks(current_block, block)
                
                if should_merge:
                    # Merge with current block
                    current_block["content"] += " " + content
                    
                    # Expand bounding box to include new block
                    current_bbox = current_block["bbox"]
                    new_bbox = block["bbox"]
                    current_block["bbox"] = [
                        min(current_bbox[0], new_bbox[0]),  # min x0
                        min(current_bbox[1], new_bbox[1]),  # min y0  
                        max(current_bbox[2], new_bbox[2]),  # max x1
                        max(current_bbox[3], new_bbox[3])   # max y1
                    ]
                    
                    current_block["spans"].append(block)
                else:
                    # Finalize current block and start new one
                    self._finalize_consolidated_block(current_block)
                    consolidated_blocks.append(current_block)
                    
                    # Start new block
                    current_block = {
                        "content": content,
                        "bbox": block["bbox"][:],
                        "page": block["page"],
                        "font": block.get("font", ""),
                        "font_size": block.get("font_size", 0),
                        "font_flags": block.get("font_flags", 0),
                        "color": block.get("color", 0),
                        "is_bold": block.get("is_bold", False),
                        "is_italic": block.get("is_italic", False),
                        "block_id": block.get("block_id", 0),
                        "spans": [block],
                        "reading_order": block.get("reading_order", 0)
                    }
        
        # Don't forget the last block
        if current_block is not None:
            self._finalize_consolidated_block(current_block)
            consolidated_blocks.append(current_block)
        
        return consolidated_blocks
    
    def _should_merge_text_blocks(self, current_block: Dict, new_block: Dict) -> bool:
        """Determine if two text blocks should be merged"""
        
        # Same block_id means they're from the same paragraph/section
        if current_block.get("block_id") == new_block.get("block_id"):
            return True
        
        # Don't merge if new block starts with definition pattern (Word followed by colon)
        new_content = new_block.get("content", "").strip()
        if self._is_definition_start(new_content):
            return False
        
        # Don't merge if current block ends with definition pattern (suggests next should be separate)
        current_content = current_block.get("content", "").strip()
        if self._is_definition_start(new_content) or current_content.endswith(":"):
            return False
        
        # Don't merge if new block starts with numbered/bulleted list
        if self._is_list_item_start(new_content):
            return False
        
        # Don't merge if there's a significant vertical gap (indicates paragraph break)
        current_bbox = current_block["bbox"]
        new_bbox = new_block["bbox"]
        vertical_gap = abs(current_bbox[3] - new_bbox[1])  # gap between current bottom and new top
        line_height = current_bbox[3] - current_bbox[1]   # height of current block
        
        # If vertical gap is more than 1.2 line heights, it's likely a new paragraph
        if vertical_gap > line_height * 1.2:
            return False
        
        # Check if font properties are similar (same style/formatting)
        font_match = (
            current_block.get("font") == new_block.get("font") and
            abs(current_block.get("font_size", 0) - new_block.get("font_size", 0)) <= 1 and
            current_block.get("is_bold") == new_block.get("is_bold") and
            current_block.get("is_italic") == new_block.get("is_italic")
        )
        
        # Check if blocks are horizontally aligned (same indentation level)
        current_left = current_bbox[0]
        new_left = new_bbox[0]
        horizontal_alignment = abs(current_left - new_left) <= 5  # 5 point tolerance
        
        # Don't merge if both blocks are bold (likely separate headings/definitions)
        both_bold = current_block.get("is_bold", False) and new_block.get("is_bold", False)
        if both_bold and not horizontal_alignment:
            return False
        
        # Merge if on same line or close consecutive lines with same formatting and alignment
        close_vertically = vertical_gap <= line_height * 1.0
        
        return font_match and close_vertically and horizontal_alignment
    
    def _is_definition_start(self, text: str) -> bool:
        """Check if text starts a definition (word/phrase followed by colon)"""
        import re
        # Pattern: Word(s) followed by colon (like "Pull-in Ploughing:", "Plant-in Ploughing:")
        definition_pattern = r'^[A-Za-z][A-Za-z\s\-]*[A-Za-z]:\s'
        return bool(re.match(definition_pattern, text))
    
    def _is_list_item_start(self, text: str) -> bool:
        """Check if text starts a list item (numbered, bulleted, etc.)"""
        import re
        # Pattern: starts with number/letter/bullet followed by space or period
        list_patterns = [
            r'^\d+[\.\)]\s',  # 1. or 1) 
            r'^[a-zA-Z][\.\)]\s',  # a. or a)
            r'^[•·▪▫‣⁃]\s',  # bullet points
            r'^[-*+]\s',  # dash, asterisk, plus bullets
            r'^\d+\s[A-Z]',  # "1 Application", "2 Terms"
        ]
        return any(re.match(pattern, text) for pattern in list_patterns)
    
    def _finalize_consolidated_block(self, block: Dict):
        """Add final metadata to consolidated block"""
        bbox = block["bbox"]
        block.update({
            "width_points": bbox[2] - bbox[0],
            "height_points": bbox[3] - bbox[1], 
            "center_x": bbox[0] + (bbox[2] - bbox[0]) / 2,
            "center_y": bbox[1] + (bbox[3] - bbox[1]) / 2,
            "char_count": len(block["content"]),
            "span_count": len(block["spans"])
        })

    def extract_text_tables_with_pymupdf(self, doc, output_dir: Path, placement_data: Dict):
        """Fallback extraction using PyMuPDF with intelligent filtering and table combination"""
        self.logger.info("Using PyMuPDF for text and table extraction with filtering...")
        
        all_text = []
        tables_dir = output_dir / "tables"
        total_tables = 0
        all_tables_info = []  # Collect all tables for combination
        
        # First pass: Extract all tables to identify their regions
        table_regions = {}
        for page_num in range(len(doc)):
            page = doc[page_num]
            tables = page.find_tables()
            
            if tables:
                table_regions[page_num + 1] = []
                for table in tables:
                    table_regions[page_num + 1].append(list(table.bbox))
        
        # Second pass: Extract text and collect table data
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_key = f"page_{page_num + 1}"
            page_text = f"\n=== PAGE {page_num + 1} ===\n"
            
            # Get page dimensions
            page_rect = page.rect
            page_width = page_rect.width
            page_height = page_rect.height
            
            # Define zones
            header_zone_bottom = page_height * 0.12
            footer_zone_top = page_height * 0.88
            
            # Get hyperlink areas for this page
            hyperlink_areas = [link["bbox"] for link in placement_data["pages"][page_key].get("hyperlinks", [])]
            
            # Enhanced text blocks extraction with spatial intelligence and filtering
            blocks = page.get_text("dict")
            raw_text_blocks = []
            filtered_lines = []
            
            for block_num, block in enumerate(blocks.get("blocks", [])):
                if "lines" in block:
                    block_bbox = block.get("bbox", [0, 0, 0, 0])
                    
                    for line_num, line in enumerate(block["lines"]):
                        line_bbox = line.get("bbox", [0, 0, 0, 0])
                        line_y = line_bbox[1]  # Top Y coordinate
                        
                        # Collect all text spans in this line
                        line_text_parts = []
                        line_content = ""
                        
                        for span_num, span in enumerate(line["spans"]):
                            span_text = span["text"].strip()
                            if span_text:
                                line_text_parts.append(span_text)
                                line_content += span_text + " "
                                
                                # Store detailed text block info
                                text_block = {
                                    "content": span_text,
                                    "bbox": span["bbox"],
                                    "page": page_num + 1,
                                    "font": span.get("font", ""),
                                    "font_size": span.get("size", 0),
                                    "font_flags": span.get("flags", 0),
                                    "color": span.get("color", 0),
                                    "block_id": block_num,
                                    "line_id": line_num,
                                    "span_id": span_num,
                                    "block_bbox": block_bbox,
                                    "line_bbox": line_bbox,
                                    "width_points": span["bbox"][2] - span["bbox"][0],
                                    "height_points": span["bbox"][3] - span["bbox"][1],
                                    "center_x": span["bbox"][0] + (span["bbox"][2] - span["bbox"][0]) / 2,
                                    "center_y": span["bbox"][1] + (span["bbox"][3] - span["bbox"][1]) / 2,
                                    "is_bold": bool(span.get("flags", 0) & 16),
                                    "is_italic": bool(span.get("flags", 0) & 2),
                                    "char_count": len(span_text),
                                    "reading_order": block_num * 1000 + line_num * 10 + span_num
                                }
                                raw_text_blocks.append(text_block)
                        
                        # Process complete line content
                        line_content = line_content.strip()
                        if not line_content:
                            continue
                        
                        # Apply intelligent filtering
                        # Skip if in header zone and looks like header content
                        if line_y < header_zone_bottom and self.is_header_content(line_content):
                            self.logger.debug(f"Skipping header content: {line_content[:50]}")
                            continue
                        
                        # Skip if in footer zone and looks like footer content  
                        if line_y > footer_zone_top and self.is_footer_content(line_content):
                            self.logger.debug(f"Skipping footer content: {line_content[:50]}")
                            continue
                        
                        # Skip if line is inside any table region
                        if self.is_line_in_table_region(line_bbox, table_regions.get(page_num + 1, [])):
                            self.logger.debug(f"Skipping table content: {line_content[:50]}")
                            continue
                        
                        # Check if line contains hyperlink and mark it
                        line_text = line_content
                        if self.line_contains_hyperlink(line_bbox, hyperlink_areas):
                            line_text = f"🔗 {line_content} [HYPERLINK]"
                        
                        filtered_lines.append(line_text)
            
            # Add filtered content to page text
            if filtered_lines:
                page_text += "\n".join(filtered_lines) + "\n"
            else:
                page_text += "[No main content text on this page]\n"
            
            all_text.append(page_text)
            
            # Store text blocks (with zone information)
            for block in raw_text_blocks:
                block["zone"] = self.get_text_zone(block["center_y"], page_height)
                block["is_hyperlink"] = self.line_contains_hyperlink(block["bbox"], hyperlink_areas)
            
            # Consolidate fragmented text blocks into coherent paragraphs
            placement_data["pages"][page_key]["text_blocks"] = raw_text_blocks
            consolidated_blocks = self.consolidate_text_blocks(placement_data["pages"][page_key])
            placement_data["pages"][page_key]["text_blocks"] = consolidated_blocks
            placement_data["pages"][page_key]["raw_text_blocks"] = raw_text_blocks  # Keep originals for reference
            
            # Extract and collect table data for later combination
            try:
                tables = page.find_tables()
                for table_num, table in enumerate(tables):
                    # Get table data with improved parsing
                    table_data = self.extract_table_improved(table)
                    table_bbox = table.bbox
                    
                    if table_data:  # Only process if we got valid data
                        # Extract table header/caption from surrounding text
                        table_header_info = self._extract_table_header(page, table_bbox, page_num + 1)
                        
                        table_info = {
                            'data': table_data,
                            'page': page_num + 1,
                            'bbox': list(table_bbox),
                            'table_object': table,  # Keep reference for more extraction if needed
                            'table_header': table_header_info.get('header', ''),  # Extract header text
                            'placement_key': self._create_placement_key(table_header_info.get('header', ''), page_num + 1),  # Create placement key
                            'content_context': table_header_info.get('context', '')  # For consolidation
                        }
                        all_tables_info.append(table_info)
                    
            except Exception as e:
                self.logger.warning(f"Table extraction failed on page {page_num + 1}: {e}")
        
        # Now combine split tables using the collected table information
        combined_tables = self.combine_split_pymupdf_tables(all_tables_info)
        
        # Save combined tables
        for table_num, table_info in enumerate(combined_tables):
            total_tables += 1
            table_id = f"table_{total_tables}"
            
            table_data = table_info['data']
            table_bbox = table_info['bbox']
            page_num = table_info['page']
            
            # Calculate table metrics
            table_width = table_bbox[2] - table_bbox[0]
            table_height = table_bbox[3] - table_bbox[1]
            table_area = table_width * table_height
            
            # Analyze table structure
            rows_count = len(table_data) if table_data else 0
            cols_count = len(table_data[0]) if table_data and table_data[0] else 0
            
            # Enhanced table placement data with spatial intelligence and header info
            table_header = table_info.get('table_header', {})
            table_placement = {
                "id": table_id,
                "bbox": table_bbox,
                "page": page_num,
                "rows": rows_count,
                "cols": cols_count,
                "confidence": 0.75,  # Lower confidence for PyMuPDF
                "width_points": table_width,
                "height_points": table_height,
                "area_points": table_area,
                "center_x": table_bbox[0] + table_width / 2,
                "center_y": table_bbox[1] + table_height / 2,
                "cell_width_avg": table_width / cols_count if cols_count > 0 else 0,
                "cell_height_avg": table_height / rows_count if rows_count > 0 else 0,
                "table_density": (rows_count * cols_count) / table_area if table_area > 0 else 0,
                "has_header": rows_count > 1,  # Assume first row is header if multiple rows
                "extraction_method": "pymupdf_filtered",
                "combined_from_pages": table_info.get('combined_from_pages', [page_num]),
                "table_header": table_header.get('header', ''),  # Table caption/title
                "content_context": table_header.get('context', ''),  # Surrounding context
                "placement_key": table_header.get('header', f'table_{total_tables}').lower().replace(' ', '_'),
                "is_combined": table_info.get('is_combined', False)
            }
            
            page_key = f"page_{page_num}"
            placement_data["pages"][page_key]["tables"].append(table_placement)
            
            # Save table data as Excel
            if table_data:
                import pandas as pd
                df = pd.DataFrame(table_data[1:], columns=table_data[0] if table_data else [])
                table_file = tables_dir / f"{table_id}_page_{page_num}.xlsx"
                df.to_excel(table_file, index=False)
                self.logger.info(f"Saved PyMuPDF table: {table_file}")
        
        # Save clean text
        clean_text = []
        for page_text in all_text:
            clean_lines = page_text.split('\n')
            # Remove page markers and empty lines for clean version
            filtered_clean_lines = []
            for line in clean_lines:
                line = line.strip()
                if line and not line.startswith('=== PAGE') and not line.startswith('[No main content'):
                    filtered_clean_lines.append(line)
            if filtered_clean_lines:
                clean_text.extend(filtered_clean_lines)
        
        # Save text with page markers
        text_file = output_dir / "text_with_pages.txt"
        with open(text_file, "w", encoding="utf-8") as f:
            f.write("\n".join(all_text))
        
        # Save clean content
        clean_file = output_dir / "clean_content_text.txt"
        with open(clean_file, "w", encoding="utf-8") as f:
            f.write("\n".join(clean_text))
        
        self.logger.info(f"Extracted {total_tables} tables using PyMuPDF with filtering")
        self.logger.info(f"Saved clean content to: {clean_file}")
        
        return placement_data
    
    def extract_tables_azure_di(self, pdf_path: Path, page_num: int) -> List[Dict]:
        """Extract tables using Azure Document Intelligence for high quality"""
        try:
            with open(pdf_path, "rb") as f:
                pdf_content = f.read()
            
            # Analyze with Azure DI
            poller = self.azure_client.begin_analyze_document(
                "prebuilt-layout", pdf_content
            )
            result = poller.result()
            
            tables = []
            for table in result.tables:
                # Check if table is on the specified page
                if hasattr(table, 'bounding_regions') and table.bounding_regions:
                    table_page = table.bounding_regions[0].page_number
                    if table_page != page_num:
                        continue
                
                # Extract table data with proper structure
                table_data = self.parse_azure_table(table)
                
                if table_data:
                    tables.append(table_data)
            
            return tables
            
        except Exception as e:
            self.logger.warning(f"Azure DI table extraction failed: {e}")
            return []
    
    def parse_azure_table(self, azure_table) -> Dict:
        """Parse Azure DI table into structured format"""
        try:
            # Get table dimensions
            rows = azure_table.row_count
            cols = azure_table.column_count
            
            # Initialize table grid
            table_grid = [['' for _ in range(cols)] for _ in range(rows)]
            
            # Fill the grid with cell content
            for cell in azure_table.cells:
                row_idx = cell.row_index
                col_idx = cell.column_index
                content = cell.content.strip() if cell.content else ''
                
                # Handle merged cells
                for r in range(row_idx, row_idx + cell.row_span):
                    for c in range(col_idx, col_idx + cell.column_span):
                        if r < rows and c < cols:
                            table_grid[r][c] = content
            
            # Calculate bounding box if available
            bbox = [0, 0, 100, 100]  # Default
            if hasattr(azure_table, 'bounding_regions') and azure_table.bounding_regions:
                polygon = azure_table.bounding_regions[0].polygon
                if len(polygon) >= 4:
                    # Handle different polygon coordinate formats
                    x_coords = []
                    y_coords = []
                    
                    for point in polygon:
                        if hasattr(point, 'x') and hasattr(point, 'y'):
                            # Point objects with x, y attributes
                            x_coords.append(point.x)
                            y_coords.append(point.y)
                        elif isinstance(point, (list, tuple)) and len(point) >= 2:
                            # List/tuple coordinates [x, y]
                            x_coords.append(point[0])
                            y_coords.append(point[1])
                        elif hasattr(point, '__getitem__') and len(point) >= 2:
                            # Array-like coordinates
                            x_coords.append(point[0])
                            y_coords.append(point[1])
                    
                    if x_coords and y_coords:
                        bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
            
            # Calculate metrics
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height
            
            return {
                "data": table_grid,
                "rows": rows,
                "cols": cols,
                "bbox": bbox,
                "width": width,
                "height": height,
                "area": area,
                "center_x": bbox[0] + width / 2,
                "center_y": bbox[1] + height / 2,
                "cell_width_avg": width / cols if cols > 0 else 0,
                "cell_height_avg": height / rows if rows > 0 else 0,
                "table_density": (rows * cols) / area if area > 0 else 0,
                "has_header": rows > 1,
                "confidence": 0.95  # High confidence for Azure DI
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to parse Azure table: {e}")
            return None
    
    def extract_table_improved(self, pymupdf_table) -> List[List[str]]:
        """Improved PyMuPDF table extraction with better text handling"""
        try:
            # Try multiple extraction approaches
            table_data = None
            
            # Method 1: Standard extraction
            try:
                table_data = pymupdf_table.extract()
            except:
                pass
            
            # Method 2: Cell-by-cell extraction if standard fails
            if not table_data or self.is_table_corrupted(table_data):
                try:
                    # Get table bbox and extract text within it
                    bbox = pymupdf_table.bbox
                    page = pymupdf_table.page
                    
                    # Extract text blocks within table area
                    text_blocks = page.get_text("dict", clip=bbox)
                    table_data = self.reconstruct_table_from_text(text_blocks, pymupdf_table)
                except:
                    pass
            
            return table_data if table_data else []
            
        except Exception as e:
            self.logger.warning(f"Improved table extraction failed: {e}")
            return []
    
    def is_table_corrupted(self, table_data: List[List[str]]) -> bool:
        """Check if table data appears corrupted"""
        if not table_data:
            return True
        
        # Check for excessive newlines or concatenated content
        for row in table_data[:5]:  # Check first few rows
            for cell in row:
                if isinstance(cell, str):
                    if cell.count('\n') > 3:  # Too many line breaks
                        return True
                    if len(cell) > 500:  # Suspiciously long cell content
                        return True
        
        return False
    
    def reconstruct_table_from_text(self, text_dict: Dict, pymupdf_table) -> List[List[str]]:
        """Reconstruct table from text blocks when standard extraction fails"""
        try:
            # This is a simplified approach - in practice, this would need
            # sophisticated spatial analysis to correctly align text into cells
            
            # For now, return the original extraction if available
            return pymupdf_table.extract()
            
        except:
            return []
    
    def save_placement_data(self, output_dir: Path, placement_data: Dict):
        """Save complete placement data to JSON"""
        placement_file = output_dir / "complete_placement_data.json"
        
        with open(placement_file, "w", encoding="utf-8") as f:
            json.dump(placement_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved placement data to: {placement_file}")
        
        # Create spatial analysis file
        self.create_spatial_analysis(output_dir, placement_data)
    
    def create_spatial_analysis(self, output_dir: Path, placement_data: Dict):
        """Create spatial analysis with element relationships"""
        spatial_analysis = {
            "document_spatial_analysis": {
                "filename": placement_data["document_info"]["filename"],
                "total_pages": placement_data["document_info"]["total_pages"]
            },
            "page_layouts": {},
            "element_relationships": {},
            "spatial_patterns": {}
        }
        
        for page_key, page_data in placement_data["pages"].items():
            page_num = int(page_key.split("_")[1])
            
            # Analyze page layout
            images = page_data.get("images", [])
            tables = page_data.get("tables", [])
            text_blocks = page_data.get("text_blocks", [])
            hyperlinks = page_data.get("hyperlinks", [])
            
            # Calculate layout metrics
            page_width = page_data["page_size"]["width"]
            page_height = page_data["page_size"]["height"]
            
            # Analyze element distribution
            spatial_analysis["page_layouts"][page_key] = {
                "page_dimensions": {"width": page_width, "height": page_height},
                "element_counts": {
                    "images": len(images),
                    "tables": len(tables),
                    "text_blocks": len(text_blocks),
                    "hyperlinks": len(hyperlinks)
                },
                "coverage_analysis": {
                    "image_coverage": sum(img.get("area_points", 0) for img in images) / (page_width * page_height) * 100,
                    "table_coverage": sum(tbl.get("area_points", 0) for tbl in tables) / (page_width * page_height) * 100,
                    "text_density": len(text_blocks) / (page_width * page_height) * 10000
                },
                "layout_zones": {
                    "header_zone": {"y_start": 0, "y_end": page_height * 0.15},
                    "body_zone": {"y_start": page_height * 0.15, "y_end": page_height * 0.85},
                    "footer_zone": {"y_start": page_height * 0.85, "y_end": page_height}
                }
            }
            
            # Analyze element relationships (proximity, alignment, etc.)
            relationships = []
            
            # Find text near images
            for img in images:
                img_center_x = img.get("center_x", 0)
                img_center_y = img.get("center_y", 0)
                
                nearby_text = []
                for text in text_blocks:
                    text_center_x = text.get("center_x", 0)
                    text_center_y = text.get("center_y", 0)
                    
                    # Calculate distance
                    distance = ((img_center_x - text_center_x) ** 2 + (img_center_y - text_center_y) ** 2) ** 0.5
                    
                    if distance < 100:  # Within 100 points
                        nearby_text.append({
                            "text_content": text["content"][:50] + "..." if len(text["content"]) > 50 else text["content"],
                            "distance": round(distance, 2),
                            "relative_position": self.get_relative_position(img, text)
                        })
                
                if nearby_text:
                    relationships.append({
                        "type": "image_text_proximity",
                        "image_id": img["id"],
                        "nearby_text": nearby_text[:5]  # Limit to 5 closest
                    })
            
            spatial_analysis["element_relationships"][page_key] = relationships
        
        # Save spatial analysis
        spatial_file = output_dir / "spatial_analysis.json"
        with open(spatial_file, "w", encoding="utf-8") as f:
            json.dump(spatial_analysis, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Created spatial analysis: {spatial_file}")
    
    def get_relative_position(self, element1: Dict, element2: Dict) -> str:
        """Determine relative position between two elements"""
        x1, y1 = element1.get("center_x", 0), element1.get("center_y", 0)
        x2, y2 = element2.get("center_x", 0), element2.get("center_y", 0)
        
        dx = x2 - x1
        dy = y2 - y1
        
        if abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        else:
            return "below" if dy > 0 else "above"
    
    def create_summary_report(self, output_dir: Path, placement_data: Dict):
        """Create a summary report of extracted content"""
        summary = {
            "extraction_summary": {
                "document": placement_data["document_info"]["filename"],
                "total_pages": placement_data["document_info"]["total_pages"],
                "total_images": sum(len(page_data["images"]) for page_data in placement_data["pages"].values()),
                "total_tables": sum(len(page_data["tables"]) for page_data in placement_data["pages"].values()),
                "total_hyperlinks": sum(len(page_data["hyperlinks"]) for page_data in placement_data["pages"].values()),
                "total_text_blocks": sum(len(page_data["text_blocks"]) for page_data in placement_data["pages"].values())
            },
            "page_breakdown": {}
        }
        
        for page_key, page_data in placement_data["pages"].items():
            summary["page_breakdown"][page_key] = {
                "images": len(page_data["images"]),
                "tables": len(page_data["tables"]),
                "hyperlinks": len(page_data["hyperlinks"]),
                "text_blocks": len(page_data["text_blocks"])
            }
        
        summary_file = output_dir / "extraction_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        
        # Print summary to console
        print(f"\n📄 EXTRACTION SUMMARY for {summary['extraction_summary']['document']}")
        print(f"📑 Pages: {summary['extraction_summary']['total_pages']}")
        print(f"🖼️  Images: {summary['extraction_summary']['total_images']}")
        print(f"📊 Tables: {summary['extraction_summary']['total_tables']}")
        print(f"🔗 Hyperlinks: {summary['extraction_summary']['total_hyperlinks']}")
        print(f"📝 Text Blocks: {summary['extraction_summary']['total_text_blocks']}")
        print(f"💾 Output saved to: {output_dir}")


def main():
    """Main execution function"""
    print("🚀 Starting Advanced PDF Extraction with Placement Tracking")
    print("=" * 60)
    
    extractor = PlacementPDFExtractor()
    extractor.process_all_pdfs()
    
    print("\n✅ Extraction completed!")
    print(f"📁 Check the 'output' folder for results")


if __name__ == "__main__":
    main()

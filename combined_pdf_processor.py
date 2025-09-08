import os
import json
import re
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def normalize_text(text):
    """Basic text normalization for comparison, ignoring numbers and punctuation."""
    text = text.lower()
    # Remove any character that is not a letter or a space
    text = re.sub(r'[^a-z\s]', '', text)
    # Collapse multiple spaces into a single space and strip leading/trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_similarity(text1, text2, vectorizer):
    """
    Calculates similarity based on two conditions:
    1. High direct sequence match (difflib), with a boost if one contains the other.
    2. High semantic similarity (cosine).
    Both must be met for a match.
    """
    norm_text1 = normalize_text(text1)
    norm_text2 = normalize_text(text2)

    if not norm_text1 or not norm_text2:
        return 0, 0  # Return scores for both methods

    # 1. Difflib for sequence matching
    difflib_score = difflib.SequenceMatcher(None, norm_text1, norm_text2).ratio()

    # Boost score if one heading is a substring of the other, to handle elaborations.
    if norm_text1 in norm_text2 or norm_text2 in norm_text1:
        difflib_score = max(difflib_score, 0.95) # Boost to ensure it passes the threshold

    # 2. Cosine similarity using TF-IDF
    try:
        tfidf_matrix = vectorizer.fit_transform([norm_text1, norm_text2])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except ValueError:
        cosine_sim = 0

    return difflib_score, cosine_sim


def structure_document(doc_content, doc_name):
    """
    Structures a document's content into a list of sections.
    Content before the first heading is grouped into a default "Introduction" section.
    """
    structured_doc = {
        "doc_name": doc_name,
        "sections": [],
        "has_headings": False
    }
    
    current_section = None
    
    HEADING_STOP_WORDS = ['note', 'caution', 'warning', 'figure', 'table']

    for item in doc_content:
        # Correct image/table paths
        if item.get('path'):
            item['path'] = os.path.join(doc_name, item['path']).replace('\\', '/')

        content = item.get('content', '').strip()
        content_lower = content.lower()
        
        # Exclude list-like items (e.g., "a.", "b.") from being headings
        is_list_item = re.match(r'^[a-zA-Z]\.\s', content)

        is_heading = (
            item.get('type') == 'paragraph'
            and item.get('role') == 'sectionHeading'
            and not any(content_lower.startswith(word) for word in HEADING_STOP_WORDS)
            and not is_list_item
        )

        if is_heading:
            structured_doc['has_headings'] = True
            # Save the previous section if it exists
            if current_section:
                structured_doc['sections'].append(current_section)
            
            # Add source document information to the heading item itself
            item['source_documents'] = [doc_name]

            # Start a new section
            current_section = {
                "heading_text": item.get('content', ''),
                "content": [item] # Start with the heading paragraph itself
            }
        else: # Not a heading
            # If this is content before the first heading, create a default section.
            if not current_section and not structured_doc['has_headings']:
                # Create a synthetic heading for the pre-heading content
                heading_item = {
                    "type": "paragraph",
                    "role": "sectionHeading",
                    "content": "Introduction",
                    "source_documents": [doc_name]
                }
                current_section = {
                    "heading_text": "Introduction",
                    "content": [heading_item]
                }
            
            # Add content to the current section
            if current_section:
                current_section['content'].append(item)
            
    # Add the last section
    if current_section:
        structured_doc['sections'].append(current_section)
        
    return structured_doc

def combine_documents(all_docs_data, difflib_threshold=0.9, cosine_threshold=0.9):
    """
    Merges multiple documents based on semantic heading similarity.
    A section is merged if BOTH difflib and cosine similarity scores meet their thresholds.
    """
    if not all_docs_data:
        return []

    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Choose the first document as the base
    base_doc = all_docs_data.pop(0)
    
    # Process other documents against the base document
    for other_doc in all_docs_data:
        # Keep track of sections from other_doc that get merged
        merged_other_sections = set()

        # First pass: merge into base document
        for other_section_idx, other_section in enumerate(other_doc['sections']):
            best_match_score = -1
            best_match_section = None
            
            # Find the best matching section in the base document
            for base_section in base_doc['sections']:
                difflib_score, cosine_score = get_similarity(base_section['heading_text'], other_section['heading_text'], vectorizer)
                
                # Check if both scores meet the threshold
                if difflib_score >= difflib_threshold and cosine_score >= cosine_threshold:
                    # Use the average of the two scores to find the *best* match among potential candidates
                    current_score = (difflib_score + cosine_score) / 2
                    if current_score > best_match_score:
                        best_match_score = current_score
                        best_match_section = base_section
            
            if best_match_section:
                print(f"  - Merging '{other_section['heading_text'][:30]}...' from '{other_doc['doc_name']}' with '{best_match_section['heading_text'][:30]}...'")
                
                # Merge content (append content of other section, skipping its heading)
                best_match_section['content'].extend(other_section['content'][1:])
                
                # Add source tracking to the heading element of the base section
                heading_element = best_match_section['content'][0]
                
                # Initialize sources with the base doc name if not present
                if 'source_documents' not in heading_element:
                    heading_element['source_documents'] = [base_doc['doc_name']]
                
                # Add the new source and keep it sorted
                sources = set(heading_element['source_documents'])
                sources.add(other_doc['doc_name'])
                heading_element['source_documents'] = sorted(list(sources))

                merged_other_sections.add(other_section_idx)

        # Second pass: add unmerged content from other_doc
        # Append unmerged sections
        for i, section in enumerate(other_doc['sections']):
            if i not in merged_other_sections:
                print(f"  - Appending section '{section['heading_text'][:30]}...' from '{other_doc['doc_name']}'")
                base_doc['sections'].append(section)

    # Flatten all sections into the final list
    combined_content = []
    for section in base_doc['sections']:
        combined_content.extend(section['content'])
        
    # Handle documents with no headings at all by appending their full content
    for doc in all_docs_data:
        if not doc['has_headings']:
            print(f"  - Appending all content from '{doc['doc_name']}' (no headings found)")
            for section in doc['sections']:
                combined_content.extend(section['content'])

    return combined_content


def main():
    """
    Main function to load, combine, and save document content.
    """
    content_folder = r"C:\Users\XJ533JH\Downloads\doc_ext\extracted_content"
    output_json_path = os.path.join(content_folder, "combined_content.json")
    
    all_docs_data = []
    
    print("--- Loading and Structuring Documents for Combination ---")
    for dir_name in os.listdir(content_folder):
        dir_path = os.path.join(content_folder, dir_name)
        if os.path.isdir(dir_path):
            json_path = os.path.join(dir_path, f"{dir_name}_unified.json")
            if os.path.exists(json_path):
                print(f"Loading: {json_path}")
                with open(json_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    structured_doc = structure_document(content, dir_name)
                    all_docs_data.append(structured_doc)

    if not all_docs_data:
        print("No structured documents to combine.")
        return

    print("\n--- Combining Documents ---")
    # Sort documents by name to have a consistent base document
    all_docs_data.sort(key=lambda x: x['doc_name'])
    combined_data = combine_documents(all_docs_data)

    print(f"\n--- Saving Combined Content to {output_json_path} ---")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)
        
    print("Combination process finished successfully.")

if __name__ == "__main__":
    main()

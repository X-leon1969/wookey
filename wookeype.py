import sys
import fitz  # PyMuPDF
import argparse
from PIL import Image
import pytesseract
from PIL import ImageEnhance, ImageFilter
import os
import sqlite3

def expand_page_range(page_spec, total_pages):
    """
    Expand a page specification into a list of page numbers.
    
    Args:
        page_spec (str): Page specification (e.g., '1-32, 35, 37', '5', 'all').
        total_pages (int): Total number of pages in the PDF.
    
    Returns:
        list: Expanded list of page numbers.
    
    Raises:
        ValueError: If the specification is invalid.
    """
    if page_spec.lower() == 'all':
        return list(range(1, total_pages + 1))
    
    page_set = set()
    parts = [p.strip() for p in page_spec.split(',')]
    for part in parts:
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                if start < 1 or end > total_pages or start > end:
                    raise ValueError(f"Invalid range '{part}' in '{page_spec}'. Must be between 1 and {total_pages} with start <= end.")
                page_set.update(range(start, end + 1))
            except ValueError as e:
                if not str(e).startswith("Invalid range"):
                    raise ValueError(f"Invalid range format '{part}' in '{page_spec}'. Use 'start-end' (e.g., '1-32').")
                raise
        else:
            try:
                page = int(part)
                if page < 1 or page > total_pages:
                    raise ValueError(f"Page number {page} in '{page_spec}' is out of range (1 to {total_pages}).")
                page_set.add(page)
            except ValueError:
                raise ValueError(f"Invalid page number '{part}' in '{page_spec}'. Must be an integer.")
    
    return sorted(list(page_set))

def extract_text_from_page(pdf_path, page_numbers, ocr=False):
    """
    Extract text from specified pages in a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file.
        page_numbers (list): List of page numbers to process (1-based).
        ocr (bool): If True, always use OCR; otherwise, use text layer with OCR fallback.
    
    Returns:
        dict: Mapping of page numbers to their extracted text.
    
    Raises:
        FileNotFoundError: If the PDF file does not exist.
    """
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        all_text = {}
        
        for page_num in page_numbers:
            if page_num < 1 or page_num > total_pages:
                continue
            page = doc[page_num - 1]  # Convert to 0-based index
            
            text = page.get_text().strip()
            if text and not ocr:
                # print(f"Text extracted from page {page_num} using text layer.")
                all_text[page_num] = f"--- Page {page_num} ---\n{text}"
            else:
                if not text:
                    print(f"No text layer found on page {page_num}, falling back to OCR.")
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img_gray = img.convert('L')
                enhancer = ImageEnhance.Contrast(img_gray)
                img_enhanced = enhancer.enhance(2.0)
                img_enhanced = img_enhanced.filter(ImageFilter.MedianFilter())
                text = pytesseract.image_to_string(img_enhanced).strip()
                if text:
                    # print(f"Text extracted from page {page_num} using OCR.")
                    all_text[page_num] = f"--- Page {page_num} ---\n{text}"
                else:
                    # print(f"No text extracted from page {page_num} using OCR.")
                    all_text[page_num] = f"--- Page {page_num} ---\n(No text available)"
        
        doc.close()
        return all_text
    
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{pdf_path}' was not found.")
    except Exception as e:
        raise Exception(f"Error processing the PDF: {str(e)}")

def get_current_registration(db_conn, pdf_name, page_number):
    """
    Retrieve the current registration for a specific page from the database.
    
    Args:
        db_conn: SQLite database connection.
        pdf_name (str): PDF filename without path.
        page_number (int): Page number (1-based).
    
    Returns:
        tuple: (page_document_number, page_text) or (None, None) if not found.
    """
    cursor = db_conn.cursor()
    cursor.execute("""
        SELECT page_document_number, page_text 
        FROM documents_assembled_pages 
        WHERE source_pdf = ? AND page_number = ?
    """, (pdf_name, page_number))
    result = cursor.fetchone()
    return result if result else (None, None)

def save_to_db(db_conn, pdf_name, page_number, doc_number, text):
    """
    Save or update the document number and text for a specific page in the database.
    
    Args:
        db_conn: SQLite database connection.
        pdf_name (str): PDF filename without path.
        page_number (int): Page number (1-based).
        doc_number (str): Document number to save.
        text (str): Extracted page text.
    """
    cursor = db_conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents_assembled_pages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_pdf TEXT,
            page_number INTEGER,
            page_document_number TEXT,
            page_text TEXT,
            UNIQUE(source_pdf, page_number)
        )
    """)
    cursor.execute("""
        INSERT OR REPLACE INTO documents_assembled_pages 
        (source_pdf, page_number, page_document_number, page_text)
        VALUES (?, ?, ?, ?)
    """, (pdf_name, page_number, doc_number, text))
    db_conn.commit()
    print(f"Saved document number '{doc_number}' for page {page_number} of '{pdf_name}' to database.")

def main():
    parser = argparse.ArgumentParser(description="wookeype.py - Extract text and optionally set document numbers for PDF pages.")
    parser.add_argument("--page", required=True, help="Page number (e.g., '5'), range (e.g., '1-32, 35, 37'), or 'all'")
    parser.add_argument("--ocr", action="store_true", help="Force OCR instead of using text layer")
    parser.add_argument("--set", nargs='?', const=True, default=False, help="Set document number: omit value to prompt, or provide value (e.g., --set '12345') to set directly")
    parser.add_argument("--db", help="Path to the SQLite database file (required with --set)")
    parser.add_argument("pdf_file", help="Path to the PDF file")
    
    args = parser.parse_args()
    
    # Validate --db requirement with --set
    if args.set and not args.db:
        print("Error: --db is required when --set is used.")
        sys.exit(1)
    
    pdf_name = os.path.basename(args.pdf_file)
    db_conn = sqlite3.connect(args.db) if args.db else None
    
    try:
        doc = fitz.open(args.pdf_file)
        total_pages = len(doc)
        doc.close()
        
        page_numbers = expand_page_range(args.page, total_pages)
        # print(f"Expanded page range: {', '.join(map(str, page_numbers))}")
        
        text_dict = extract_text_from_page(args.pdf_file, page_numbers, args.ocr)
        
        if args.set:
            if isinstance(args.set, str):  # --set has a value
                doc_number = args.set
                for page_num in page_numbers:
                    page_text = text_dict.get(page_num, f"--- Page {page_num} ---\n(No text available)")
                    save_to_db(db_conn, pdf_name, page_num, doc_number, page_text)
            else:  # --set is True (no value provided)
                for page_num in page_numbers:
                    current_doc_number, current_text = get_current_registration(db_conn, pdf_name, page_num)
                    print(f"\nCurrent registration for page {page_num} of '{pdf_name}':")
                    if current_doc_number:
                        print(f"  Document Number: {current_doc_number}")
                        print(f"  Text: {current_text}")
                    else:
                        print(f"  No registration found in database.")
                    
                    page_text = text_dict.get(page_num, f"--- Page {page_num} ---\n(No text available)")
                    print(f"\nText for page {page_num} of '{args.pdf_file}':\n")
                    print(page_text)
                    
                    doc_number = input(f"Enter document number for page {page_num} (or press Enter to exit): ").strip()
                    if not doc_number:
                        print("No input provided. Exiting.")
                        if db_conn:
                            db_conn.close()
                        sys.exit(0)
                    
                    save_to_db(db_conn, pdf_name, page_num, doc_number, page_text)
        else:
            page_range_str = args.page if args.page.lower() == 'all' else ','.join(map(str, page_numbers))
            print(f"\nText extracted from '{args.pdf_file}' (pages {page_range_str}):\n")
            # for page_num in page_numbers:
                # print(text_dict.get(page_num, f"--- Page {page_num} ---\n(No text available)"))
            
            base_name = os.path.splitext(args.pdf_file)[0]
            output_file = f"{base_name}.pdf.{page_range_str}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("\n\n".join(text_dict.get(pn, f"--- Page {pn} ---\n(No text available)") for pn in page_numbers))
            print(f"\nText saved to '{output_file}'.")
    
    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"Error: {str(e)}")
        if db_conn:
            db_conn.close()
        sys.exit(1)
    
    if db_conn:
        db_conn.close()

if __name__ == "__main__":
    main()

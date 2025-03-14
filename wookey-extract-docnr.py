import sys
import fitz  
from PIL import Image
import pytesseract
import re
import itertools
from operator import itemgetter
import time
from datetime import datetime
import os
import sqlite3
from datetime import datetime
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob  
import math

# Global variable to control output
quiet_mode = False
VERBOSE_MODE = False

def print_if_not_quiet(*args, **kwargs):
    """Print only if not in quiet mode."""
    if not quiet_mode:
        print(*args, **kwargs)

def format_size(bytes_size):
    """Convert bytes to human-readable format (e.g., KB, MB, GB)."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.2f} TB"

def format_time(seconds):
    """Convert seconds into a human-readable format."""
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes = seconds // 60
    seconds_remainder = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m" + (f"{seconds_remainder}s" if seconds_remainder > 0 else "")
    hours = minutes // 60
    minutes_remainder = minutes % 60
    return f"{hours}h" + (f"{minutes_remainder}m" if minutes_remainder > 0 else "")

def extract_document_number(page, page_num, dpi, length=None, previous_number=None, quick=False, ocr=False):
    """Extract document number from the page, prioritizing first and last lines, with optional quick mode.
    
    Args:
        page: The PDF page object.
        page_num: The page number (for logging).
        dpi: DPI for OCR image conversion.
        length (int, optional): The exact length of the digit sequence to match (e.g., 4 for exactly 4 digits).
        previous_number (str, optional): The document number from the previous page for range checking and structure matching.
        quick (bool, optional): If True, only check the first and last lines and match the structure of the previous number.
    
    Returns:
        tuple: (doc_number, alltext) where doc_number is the extracted document number or None.
    """
    from PIL import ImageEnhance, ImageFilter, ImageOps, Image

    # Initialize variables
    alltext = ""
    doc_number = None

    # In quick mode, first try to extract text using page.get_text()
    if quick:
        alltext = page.get_text("text").strip()
        if ocr or not alltext:
            # If no text is extracted, fall back to OCR
            if VERBOSE_MODE:
                print_if_not_quiet(f"Quick mode: No text extracted from page {page_num + 1} using get_text, falling back to OCR")
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            width, height = img.size
            
            # Step 1: Convert to grayscale
            img_gray = img.convert('L')
            
            # Step 2: Enhance contrast
            enhancer = ImageEnhance.Contrast(img_gray)
            img_enhanced = enhancer.enhance(2.0)  # Enhance contrast
            img_enhanced = img_enhanced.filter(ImageFilter.MedianFilter())  # Reduce noise
            
            # Step 3: Read text using OCR
            alltext = pytesseract.image_to_string(img_enhanced)
    else:
        # Non-quick mode: always use OCR
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        width, height = img.size
        
        # Step 1: Convert to grayscale
        img_gray = img.convert('L')
        
        # Step 2: Enhance contrast
        enhancer = ImageEnhance.Contrast(img_gray)
        img_enhanced = enhancer.enhance(2.0)  # Enhance contrast
        img_enhanced = img_enhanced.filter(ImageFilter.MedianFilter())  # Reduce noise
        
        # Step 3: Read all text from the page
        alltext = pytesseract.image_to_string(img_enhanced)

    # Step 4: Check for document number, prioritizing first and last lines
    lines = [line.strip() for line in alltext.split('\n') if line.strip()]
    if not lines:
        return None, alltext
    
    first_line = lines[0]
    if len(first_line) < 3:
        first_line = lines[1]
        if len(first_line) < 3 or ( len(first_line) == 3 and first_line.endswith('.') ):
            first_line = lines[2]
    else:
        if first_line == '10.2.e Doc. 1':
            first_line = 'Doc. 1'
        elif first_line == '10.2.e':
            first_line = lines[1]
    
    first_lines = lines[:3]
    
    if VERBOSE_MODE:
        print('fl', first_line)
        print('fls', first_lines)
        
    last_line = lines[-1]
    
    # exception 1.
    if last_line == "De volgende 6 pagina’'s zijn verwijderd i.v.m. dubbel.":
        last_line = lines[-2]
        if last_line == "":
            last_line = lines[-3]
            
    last_lines = lines[-3:] if len(lines) >= 3 else lines
    bottom_line = last_line  # Already checking the last line explicitly
    
    if VERBOSE_MODE:
        print('ll',last_line)
        print('lls',last_lines)
    # Determine the pattern of the previous number
    def get_number_pattern(number):
        if not number:
            return None
        if re.match(r'^\d+$', number):
            return "pure_digits"
        elif re.match(r'^[A-Za-z]\.\d+$', number):
            return "letter_period_digits"
        elif re.match(r'^[A-Za-z]+\.\s*\d+$', number):
            return "letters_period_digits"
        return None

    previous_pattern = get_number_pattern(previous_number) if previous_number else None
    
    # Helper function to check a single line for a document number
    def check_line(line, source_description, prefer_pattern=None):
        range_check = 200000  # Moved inside the function for clarity
        # Pattern 1: Pure digits (e.g., 1234)
        if length is not None:
            digit_pattern = rf'\b\d{{{length}}}\b'  # e.g., \b\d{4}\b for length=4
        else:
            digit_pattern = r'\b\d+\b'  # Any length if not specified
        match1 = re.search(digit_pattern, line)
        
        # Pattern 2: Letter.period.digits (e.g., D.114)
        match2 = re.search(r'\b[A-Za-z]\.\d+\b', line)
        
        # Pattern 3: Letters.period.digits (e.g., Doc. 1)
        match3 = re.search(r'\b[A-Za-z]+\.\s*\d+\b', line)

        matches = []
        if match1 and any(c.isdigit() for c in match1.group()):
            doc_number = match1.group()
            if previous_number and previous_number.isdigit() and doc_number.isdigit():
                prev_num = int(previous_number)
                curr_num = int(doc_number)
                if abs(curr_num - prev_num) > range_check:
                    print_if_not_quiet(f"Step 4: Discarding document number '{doc_number}' (outside range {range_check} of previous {previous_number}, page {page_num})")
                    return None
            matches.append(("pure_digits", doc_number))
        
        if match2 and any(c.isdigit() for c in match2.group()):
            doc_number = match2.group()
            digit_part = doc_number.split('.')[1]
            if length is None or len(digit_part) == length:
                matches.append(("letter_period_digits", doc_number))
        
        if match3 and any(c.isdigit() for c in match3.group()):
            doc_number = match3.group()
            digit_part = re.search(r'\d+', doc_number).group()
            if length is None or len(digit_part) == length:
                matches.append(("letters_period_digits", doc_number))
        
        if not matches:
            return None
        
        # If prefer_pattern is specified, prioritize the match with the same pattern
        if prefer_pattern:
            for pattern, doc_number in matches:
                if pattern == prefer_pattern:
                    if VERBOSE_MODE:
                        print_if_not_quiet(f"Step 4: Identified document number '{doc_number}' (pattern {pattern} matches previous, in {source_description}, page {page_num})")
                    return doc_number
        
        # Otherwise, return the first match
        doc_number = matches[0][1]
        if VERBOSE_MODE:
            print_if_not_quiet(f"Step 4: Identified document number '{doc_number}' (first match in {source_description}, page {page_num})")
        return doc_number

    # Check first line
    doc_number = check_line(first_line, "first line", prefer_pattern=previous_pattern)
    if doc_number:
        return doc_number, alltext
    
    # Check last line
    doc_number = check_line(last_line, "last line", prefer_pattern=previous_pattern)
    if doc_number:
        return doc_number, alltext
    
    # If quick mode is enabled, stop here
    if quick:
        return None, alltext
    
    # Otherwise, proceed with the existing checks (last three lines, bottom line, red box detection)
    # Check last 3 lines
    for line in last_lines:
        doc_number = check_line(line, "last lines")
        if doc_number:
            return doc_number, alltext
    
    # Check the very bottom line (already checked in last line, but kept for consistency)
    doc_number = check_line(bottom_line, "bottom line")
    if doc_number:
        return doc_number, alltext
    
    # If no match in first, last, or bottom line, proceed with cropping for red box detection
    box_width_right, box_height = 500, 200
    cropped_right = img_enhanced.crop((width - box_width_right, 0, width, box_height))
    
    # Preprocess for red box detection
    cropped_right = ImageOps.autocontrast(cropped_right)
    cropped_right = cropped_right.point(lambda x: 0 if x < 120 else 255)  # Adjusted threshold for red box
    cropped_right = cropped_right.filter(ImageFilter.MedianFilter(size=3))  # Stronger noise reduction
    
    # Extract text from top-right
    text_right = pytesseract.image_to_string(cropped_right)
    lines_right = [line.strip() for line in text_right.split('\n') if line.strip()]
    
    # Step 5a: Check for 6 or more consecutive digits in top-right
    range_check = 200000  # Moved inside the function for clarity
    if length is None:
        digit_pattern = r'\b\d{6,}\b'
    else:
        digit_pattern = rf'\b\d{{{length}}}\b'
    for line in lines_right:
        match = re.search(digit_pattern, line)
        if match:
            doc_number = match.group()
            if previous_number and previous_number.isdigit() and doc_number.isdigit():
                prev_num = int(previous_number)
                curr_num = int(doc_number)
                if abs(curr_num - prev_num) > range_check:
                    if VERBOSE_MODE:
                        print_if_not_quiet(f"Step 5a: Discarding document number '{doc_number}' (outside range {range_check} of previous {previous_number}, page {page_num})")
                    continue
            if VERBOSE_MODE:
                print_if_not_quiet(f"Step 5a: Identified document number '{doc_number}' (digits in top-right, page {page_num})")
            return doc_number, alltext
    
    # Step 5b: Check for letter-period-digit patterns in top-right
    for line in lines_right:
        match1 = re.search(r'\b[A-Za-z]\.\d+\b', line)
        match2 = re.search(r'\b[A-Za-z]\d+\.\d+\b', line)
        match3 = re.search(r'\b[A-Za-z]+\.\s*\d+\b', line)
        if match1 and any(c.isdigit() for c in match1.group()):
            doc_number = match1.group()
            digit_part = doc_number.split('.')[1]
            if length is None or len(digit_part) == length:
                if VERBOSE_MODE:
                    print_if_not_quiet(f"Step 5b: Identified document number '{doc_number}' (letter.period.digits in top-right, page {page_num})")
                return doc_number, alltext
        elif match2 and any(c.isdigit() for c in match2.group()):
            doc_number = match2.group()
            digit_part = re.search(r'\d+$', doc_number).group()
            if length is None or len(digit_part) == length:
                if VERBOSE_MODE:
                    print_if_not_quiet(f"Step 5b: Identified document number '{doc_number}' (letterdigit.period.digits in top-right, page {page_num})")
                return doc_number, alltext
        elif match3 and any(c.isdigit() for c in match3.group()):
            doc_number = match3.group()
            digit_part = re.search(r'\d+', doc_number).group()
            if length is None or len(digit_part) == length:
                if VERBOSE_MODE:
                    print_if_not_quiet(f"Step 5b: Identified document number '{doc_number}' (letters.period.digits in top-right, page {page_num})")
                return doc_number, alltext
    
    # Step 5c: Check for single digit or small number in top-right
    if length is None:
        digit_pattern = r'\b\d{1,5}\b'
    else:
        digit_pattern = rf'\b\d{{{length}}}\b'
    for line in lines_right:
        match = re.search(digit_pattern, line)
        if match and any(c.isdigit() for c in match.group()):
            doc_number = match.group()
            if previous_number and previous_number.isdigit() and doc_number.isdigit():
                prev_num = int(previous_number)
                curr_num = int(doc_number)
                if abs(curr_num - prev_num) > range_check:
                    if VERBOSE_MODE:
                        print_if_not_quiet(f"Step 5c: Discarding document number '{doc_number}' (outside range {range_check} of previous {previous_number}, page {page_num})")
                    continue
            if VERBOSE_MODE:
                print_if_not_quiet(f"Step 5c: Identified document number '{doc_number}' (digits in top-right, page {page_num})")
            return doc_number, alltext
    
    return None, alltext
    
def pages_to_ranges(pages):
    """Convert a list of page numbers into a string of ranges."""
    if not pages:
        return ""
    ranges = []
    for k, g in itertools.groupby(enumerate(sorted(pages)), lambda x: x[0] - x[1]):
        group = list(map(itemgetter(1), g))
        range_str = f"{group[0]}-{group[-1]}" if len(group) > 1 else str(group[0])
        ranges.append(range_str)
    return ", ".join(ranges)

def get_processed_pages(db_conn, pdf_name):
    """Get the number of processed pages for a PDF from the database."""
    cursor = db_conn.cursor()
    cursor.execute("SELECT DISTINCT page_number FROM documents_assembled_pages WHERE source_pdf = ?", (pdf_name,))
    pages = {row[0] for row in cursor.fetchall()}
    return len(pages), pages

def process_pdf(pdf_path, dpi, db_name, extract_text=False, force=False, length=None, quick=False, ocr=False, workers=1, page_range=None):
    """Process a PDF to extract document numbers from specified pages, using multiple workers for parallel processing."""
    pdf_name = os.path.basename(pdf_path)
    print_if_not_quiet(f"Processing PDF: {pdf_name}")
    
    # Create a new database connection for this thread
    db_conn = sqlite3.connect(db_name)
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print_if_not_quiet(f"Opened PDF with {total_pages} pages")
    
    # Check database for existing processing
    db_page_count, processed_pages = get_processed_pages(db_conn, pdf_name)
    if not force and db_page_count == total_pages and page_range is None:
        print_if_not_quiet(f"Skipping '{pdf_name}': Already fully processed ({total_pages} pages in DB)")
        doc.close()
        db_conn.close()
        return None, None, 0, 0
    
    # Determine pages to process
    pages_to_process = page_range if page_range is not None else list(range(1, total_pages + 1))
    print_if_not_quiet(f"Processing pages: {pages_to_process}")
    
    # Split pages into chunks based on the number of workers
    pages_per_worker = max(1, math.ceil(len(pages_to_process) / workers))
    page_chunks = [pages_to_process[i:i + pages_per_worker] for i in range(0, len(pages_to_process), pages_per_worker)]
    print_if_not_quiet(f"Dividing {len(pages_to_process)} pages into {len(page_chunks)} chunks for {workers} workers")

    doc_numbers = {}
    full_text = {}
    previous_number = None
    
    def process_page_chunk(chunk):
        """Process a chunk of specific pages and return document numbers and full text."""
        local_doc_numbers = {}
        local_full_text = {}
        local_previous_number = previous_number
        
        for page_num in chunk:
            if page_num < 1 or page_num > total_pages:
                continue
            page_idx = page_num - 1  # Convert to 0-based indexing for fitz
            page = doc[page_idx]
            doc_number, page_text = extract_document_number(page, page_idx, dpi, length=length, previous_number=local_previous_number, quick=quick, ocr=ocr)
            
            if doc_number:
                if local_previous_number and len(doc_number) < len(local_previous_number):
                    nstr = '(new nr smaller in size than previous page, reverting)'
                    doc_number = local_previous_number
                local_doc_numbers[page_num] = doc_number
                if extract_text:
                    local_full_text[page_num] = page_text
                print_if_not_quiet(f"{pdf_name} # page {page_num} # {doc_number} # {nstr if 'nstr' in locals() else ''}")
                local_previous_number = doc_number
                nstr = ''
            else:
                if local_previous_number:
                    doc_number = local_previous_number
                    local_doc_numbers[page_num] = doc_number
                    print_if_not_quiet(f"{pdf_name} # page {page_num}: No number found, using previous number {local_previous_number}")
                else:
                    print_if_not_quiet(f"{pdf_name} # page {page_num}: No document number pattern found in OCR data, using 'n/a'")
                    local_doc_numbers[page_num] = "n/a"
        
        return local_doc_numbers, local_full_text, local_previous_number

    # Process page chunks in parallel
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_chunk = {executor.submit(process_page_chunk, chunk): chunk for chunk in page_chunks}
        
        for future in as_completed(future_to_chunk):
            chunk = future_to_chunk[future]
            try:
                local_doc_numbers, local_full_text, last_number = future.result()
                # Update global results
                doc_numbers.update(local_doc_numbers)
                full_text.update(local_full_text)
                # Update previous_number for the next chunk
                previous_number = last_number
                print_if_not_quiet(f"Processed pages {chunk} of '{pdf_name}'")
            except Exception as e:
                print_if_not_quiet(f"Failed to process pages {chunk} of '{pdf_name}': {str(e)}")

    doc.close()
    size = os.path.getsize(pdf_path)
    unique_numbers = len(set(doc_numbers.values()))
    print_if_not_quiet(f"Finished processing '{pdf_name}' - Pages: {len(pages_to_process)}, Size: {format_size(size)}, Unique doc numbers: {unique_numbers}")
    
    # Save results to database
    if doc_numbers:
        save_to_db(db_conn, pdf_name, doc_numbers, full_text)
    
    db_conn.close()
    return doc_numbers, full_text, len(doc_numbers), size
    
def save_to_db(db_conn, pdf_name, doc_numbers, full_text):
    """Save page data to documents_assembled_pages, linking to documents_assembled."""
    print_if_not_quiet(f"Saving data for '{pdf_name}' to database")
    cursor = db_conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents_assembled_pages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_pdf TEXT,
            page_number INTEGER,
            page_document_number TEXT,
            page_text TEXT,
            FOREIGN KEY (source_pdf) REFERENCES documents_assembled(name)
        )
    """)
    cursor.execute("DELETE FROM documents_assembled_pages WHERE source_pdf = ?", (pdf_name,))
    
    for page_num, doc_num in doc_numbers.items():
        text = full_text.get(page_num, '') if full_text else ''
        cursor.execute("""
            INSERT INTO documents_assembled_pages (source_pdf, page_number, page_document_number, page_text)
            VALUES (?, ?, ?, ?)
        """, (pdf_name, page_num, doc_num, text))
    
    db_conn.commit()
    print_if_not_quiet(f"Database commit completed for '{pdf_name}' - {len(doc_numbers)} pages saved")

def resolve_folder_path(folder_string):
    """
    Resolve a folder string that may contain wildcards into the actual folder path (excluding filenames).
    
    Args:
        folder_string (str): A folder path or pattern, possibly containing wildcards (*, ?, [seq]).
        
    Returns:
        str or None: The resolved folder path as a string, or None if no matching path exists.
        
    Raises:
        ValueError: If the input is not a string or is empty.
    """
    # Validate input
    if not isinstance(folder_string, str):
        raise ValueError("folder_string must be a string")
    if not folder_string.strip():
        raise ValueError("folder_string cannot be empty")
    
    # Normalize the path
    folder_string = os.path.normpath(folder_string)
    
    # Split the path into directory and filename components
    dir_part, file_part = os.path.split(folder_string)
    
    # Check if the file part contains wildcards
    has_wildcards = bool(re.search(r'[\*\?\[\]]', file_part))
    
    # If the file part has wildcards, we only care about the directory part
    if has_wildcards:
        path_to_resolve = dir_part
    else:
        # If the path points to a file, use the directory part
        if os.path.isfile(folder_string):
            path_to_resolve = dir_part
        else:
            path_to_resolve = folder_string
    
    # Use glob to resolve the directory path (in case dir_part contains wildcards)
    resolved_dirs = glob.glob(path_to_resolve)
    
    # If no matching directories are found, return None
    if not resolved_dirs:
        return None
    
    # Take the first resolved directory (there should typically be only one for a directory path)
    resolved_dir = os.path.normpath(resolved_dirs[0])
    
    # Verify that the resolved path is a directory
    if not os.path.isdir(resolved_dir):
        return None
    
    return resolved_dir

def scan_folder_and_list(folder, db_conn, force=False):
    """Scan folder for PDFs, gather metadata, save to DB, and return sorted list by creation date.
    The folder is leading: removes stale entries from documents_assembled before processing."""
    print_if_not_quiet(f"Checking input: {folder}")
    
    # folder might contain wild cards
    folder_path = resolve_folder_path(folder)
    
    def parse_pdf_timestamp(pdf_ts):
        """Convert PDF timestamp (e.g., 'D:20230504112810+02'00'') to Unix timestamp."""
        if not pdf_ts or not pdf_ts.startswith('D:'):
            return None
        try:
            # Remove 'D:' prefix and handle timezone
            ts = pdf_ts[2:]
            if '+' in ts or '-' in ts:
                dt_str, tz = ts.split('+') if '+' in ts else ts.split('-')
                tz_sign = 1 if '+' in ts else -1
                tz_hours, tz_minutes = int(tz[:2]), int(tz[3:5])
                tz_offset = tz_sign * (tz_hours * 3600 + tz_minutes * 60)
            else:
                dt_str, tz_offset = ts, 0
            # Parse YYYYMMDDHHMMSS
            dt = datetime.strptime(dt_str[:14], '%Y%m%d%H%M%S')
            return int(dt.timestamp() - tz_offset)
        except (ValueError, IndexError):
            return None
    
    if os.path.isfile(folder) and folder.lower().endswith('.pdf'):
        print_if_not_quiet(f"Single PDF detected: {folder}. Processing with implied --force")
        pdf_data = []
        total_per_code = {}
        cursor = db_conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents_assembled (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                size INTEGER,
                page_count INTEGER,
                author TEXT,
                title TEXT,
                creation_date TEXT,
                modification_date TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents_assembled_pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_pdf TEXT,
                page_number INTEGER,
                page_document_number TEXT,
                page_text TEXT,
                FOREIGN KEY (source_pdf) REFERENCES documents_assembled(name)
            )
        """)
        db_conn.commit()
        
        pdf = os.path.basename(folder)
        filepath = folder
        size = os.path.getsize(filepath)
        creation_time = os.path.getctime(filepath)
        
        try:
            doc = fitz.open(filepath)
            pages = len(doc)
            metadata = doc.metadata
            print_if_not_quiet(f"Metadata - Pages: {pages}, Size: {format_size(size)}, Creation Time: {datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')}, Author: {metadata.get('author')}, Title: {metadata.get('title')}")
            doc.close()
        except (fitz.FileDataError, Exception) as e:
            errors_folder = os.path.join(os.path.dirname(folder), 'errors')
            os.makedirs(errors_folder, exist_ok=True)
            print_if_not_quiet(f"Error opening '{pdf}': {str(e)}. Moving to {errors_folder}")
            error_path = os.path.join(errors_folder, pdf)
            os.rename(filepath, error_path)
            print_if_not_quiet(f"Moved '{pdf}' to '{error_path}'")
            return [], {}, folder_path
        
        creation_timestamp = parse_pdf_timestamp(metadata.get('creationDate')) or int(creation_time)
        cursor.execute("""
            INSERT OR REPLACE INTO documents_assembled 
            (name, size, page_count, author, title, creation_date, modification_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (pdf, size, pages, metadata.get('author'), metadata.get('title'),
              str(creation_timestamp), metadata.get('modDate')))
        db_conn.commit()
        print_if_not_quiet(f"Saved and committed metadata for {pdf} to database")
        
        pdf_data.append((pdf, size, pages, creation_timestamp))
        
        code = pdf.split(' ')[0]
        if len(code) == 32 and re.match(r'^[0-9a-fA-F]{32}$', code):
            total_per_code[code] = 1
            print_if_not_quiet(f"Detected 32-char code '{code}' in filename")
        
        print_if_not_quiet(f"Prepared single PDF {pdf} for processing")
        return [(pdf, size, pages) for pdf, size, pages, _ in pdf_data], total_per_code, folder_path
    
    print_if_not_quiet(f"Expanding input: {folder}")
    if '*' in folder or '?' in folder or '[' in folder:
        pdf_files = [os.path.basename(f) for f in glob.glob(folder) if f.lower().endswith('.pdf')]
        folder_path = os.path.dirname(folder) or os.getcwd()
        print_if_not_quiet(f"Found {len(pdf_files)} PDF files matching pattern in {folder_path}")
    else:
        print_if_not_quiet(f"Scanning folder: {folder}")
        pdf_files = [f for f in os.listdir(folder) if f.lower().endswith('.pdf')]
        folder_path = folder
    print_if_not_quiet(f"Found {len(pdf_files)} PDF files in scope")
    
    pdf_data = []
    total_per_code = {}
    cursor = db_conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents_assembled (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            size INTEGER,
            page_count INTEGER,
            author TEXT,
            title TEXT,
            creation_date TEXT,
            modification_date TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents_assembled_pages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_pdf TEXT,
            page_number INTEGER,
            page_document_number TEXT,
            page_text TEXT,
            FOREIGN KEY (source_pdf) REFERENCES documents_assembled(name)
        )
    """)
    db_conn.commit()
    
    errors_folder = os.path.join(folder_path, 'errors')
    os.makedirs(errors_folder, exist_ok=True)
    print_if_not_quiet(f"Ensured errors folder exists at: {errors_folder}")
    
    # Get current PDFs in documents_assembled
    cursor.execute("SELECT name, size, page_count, creation_date FROM documents_assembled")
    assembled_pdfs_data = {}
    for row in cursor.fetchall():
        try:
            creation_time = float(row[3]) if row[3] else None
        except ValueError:
            # Handle legacy string timestamps
            creation_time = parse_pdf_timestamp(row[3]) or None
        assembled_pdfs_data[row[0].strip().lower()] = (row[1], row[2], creation_time)
    assembled_pdfs = set(assembled_pdfs_data.keys())
    print_if_not_quiet(f"Found {len(assembled_pdfs)} PDFs in documents_assembled before cleanup")
    
    # Delete records from documents_assembled that no longer exist in the folder
    pdf_files_set = {pdf.strip().lower() for pdf in pdf_files}
    stale_pdfs = assembled_pdfs - pdf_files_set
    if stale_pdfs:
        print_if_not_quiet(f"Removing {len(stale_pdfs)} stale entries from documents_assembled: {', '.join(sorted(stale_pdfs))}")
        for stale_pdf in stale_pdfs:
            cursor.execute("DELETE FROM documents_assembled WHERE name = ?", (stale_pdf,))
        db_conn.commit()
        print_if_not_quiet(f"Stale entries removed and committed")

    # Refresh assembled_pdfs after cleanup
    cursor.execute("SELECT name, size, page_count, creation_date FROM documents_assembled")
    assembled_pdfs_data = {}
    for row in cursor.fetchall():
        try:
            creation_time = float(row[3]) if row[3] else None
        except ValueError:
            # Handle legacy string timestamps
            creation_time = parse_pdf_timestamp(row[3]) or None
        assembled_pdfs_data[row[0].strip().lower()] = (row[1], row[2], creation_time)
    assembled_pdfs = set(assembled_pdfs_data.keys())
    print_if_not_quiet(f"Found {len(assembled_pdfs)} PDFs in documents_assembled after cleanup")
    
    cursor.execute("SELECT DISTINCT source_pdf FROM documents_assembled_pages")
    processed_pdfs = {row[0].strip().lower() for row in cursor.fetchall()}
    print_if_not_quiet(f"Found {len(processed_pdfs)} PDFs in documents_assembled_pages")
    
    try:
        n = 0
        t = len(pdf_files)
        for pdf in pdf_files:
            n+=1
            filepath = os.path.join(folder_path, pdf)
            pdf_normalized = pdf.strip().lower()
            
            if pdf_normalized in assembled_pdfs and not force:
                # Use existing metadata from documents_assembled
                size, pages, creation_time = assembled_pdfs_data[pdf_normalized]
                if size is None or pages is None or creation_time is None:
                    # Fallback to file system if database metadata is incomplete
                    print_if_not_quiet(f"({n}/{t}) Checking file: {pdf} (incomplete metadata)")
                    size = os.path.getsize(filepath)
                    creation_time = os.path.getctime(filepath)
                    try:
                        doc = fitz.open(filepath)
                        pages = len(doc)
                        metadata = doc.metadata
                        print_if_not_quiet(f"Metadata - Pages: {pages}, Size: {format_size(size)}, Creation Time: {datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')}, Author: {metadata.get('author')}, Title: {metadata.get('title')}")
                        doc.close()
                    except (fitz.FileDataError, Exception) as e:
                        print_if_not_quiet(f"Error opening '{pdf}': {str(e)}. Moving to {errors_folder}")
                        error_path = os.path.join(errors_folder, pdf)
                        os.rename(filepath, error_path)
                        print_if_not_quiet(f"Moved '{pdf}' to '{error_path}'")
                        continue
                    creation_timestamp = parse_pdf_timestamp(metadata.get('creationDate')) or int(creation_time)
                    cursor.execute("""
                        INSERT OR REPLACE INTO documents_assembled 
                        (name, size, page_count, author, title, creation_date, modification_date)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (pdf, size, pages, metadata.get('author'), metadata.get('title'),
                          str(creation_timestamp), metadata.get('modDate')))
                    db_conn.commit()
                    print_if_not_quiet(f"Updated incomplete metadata for {pdf} in database")
                else:
                    print_if_not_quiet(f"Using existing metadata for '{pdf}' from documents_assembled")
            else:
                # Check file for PDFs without metadata or when forced
                print_if_not_quiet(f"Checking file: {pdf}")
                size = os.path.getsize(filepath)
                creation_time = os.path.getctime(filepath)
                try:
                    doc = fitz.open(filepath)
                    pages = len(doc)
                    metadata = doc.metadata
                    print_if_not_quiet(f"Metadata - Pages: {pages}, Size: {format_size(size)}, Creation Time: {datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')}, Author: {metadata.get('author')}, Title: {metadata.get('title')}")
                    doc.close()
                except (fitz.FileDataError, Exception) as e:
                    print_if_not_quiet(f"Error opening '{pdf}': {str(e)}. Moving to {errors_folder}")
                    error_path = os.path.join(errors_folder, pdf)
                    os.rename(filepath, error_path)
                    print_if_not_quiet(f"Moved '{pdf}' to '{error_path}'")
                    continue
                
                creation_timestamp = parse_pdf_timestamp(metadata.get('creationDate')) or int(creation_time)
                cursor.execute("""
                    INSERT OR REPLACE INTO documents_assembled 
                    (name, size, page_count, author, title, creation_date, modification_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (pdf, size, pages, metadata.get('author'), metadata.get('title'),
                      str(creation_timestamp), metadata.get('modDate')))
                db_conn.commit()
                print_if_not_quiet(f"Saved and committed metadata for {pdf} to database")
                # Refresh assembled_pdfs_data
                cursor.execute("SELECT name, size, page_count, creation_date FROM documents_assembled")
                assembled_pdfs_data = {}
                for row in cursor.fetchall():
                    try:
                        creation_time = float(row[3]) if row[3] else None
                    except ValueError:
                        creation_time = parse_pdf_timestamp(row[3]) or None
                    assembled_pdfs_data[row[0].strip().lower()] = (row[1], row[2], creation_time)
                assembled_pdfs = set(assembled_pdfs_data.keys())
            
            # Add to pdf_data with the determined values
            pdf_data.append((pdf, size, pages, creation_time))
            
            code = pdf.split(' ')[0]
            if len(code) == 32 and re.match(r'^[0-9a-fA-F]{32}$', code):
                total_per_code[code] = total_per_code.get(code, 0) + 1
                print_if_not_quiet(f"Detected 32-char code '{code}' in filename, count now: {total_per_code[code]}")
    
    except KeyboardInterrupt:
        print_if_not_quiet("\nReceived Ctrl+C, stopping document processing...")
        # Commit any pending changes to ensure database consistency
        db_conn.commit()
        print_if_not_quiet("Database changes committed")
        # Return what’s processed so far
        print_if_not_quiet(f"Processed {len(pdf_data)} PDFs before interruption")
        return [(pdf, size, pages) for pdf, size, pages, _ in pdf_data], total_per_code, folder_path
    
    print_if_not_quiet(f"Folder scan completed. Processed {len(pdf_data)} PDFs. Total codes detected: {len(total_per_code)}")
    
    if not pdf_data:
        unprocessed_pdfs = [pdf for pdf in pdf_files if pdf.strip().lower() in assembled_pdfs and pdf.strip().lower() not in processed_pdfs]
        if unprocessed_pdfs:
            print_if_not_quiet("\nNo new PDFs found. Listing PDFs in folder not yet in documents_assembled_pages:")
            for i, pdf in enumerate(sorted(unprocessed_pdfs), 1):
                ctime = os.path.getctime(os.path.join(folder_path, pdf))
                size = os.path.getsize(os.path.join(folder_path, pdf))
                print_if_not_quiet(f"{i}. {pdf} - Size: {format_size(size)}, Created: {datetime.fromtimestamp(ctime).strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print_if_not_quiet("\nNo new PDFs found, and all PDFs in folder are either processed or invalid.")
    
    pdf_data.sort(key=lambda x: x[3])
    print_if_not_quiet(f"Sorted {len(pdf_data)} PDFs by creation date (ascending)")
    
    with open('pdf_list.txt', 'w') as f:
        for i, (pdf, size, pages, ctime) in enumerate(pdf_data, 1):
            f.write(f"{i}. {pdf} - {pages} pages, {format_size(size)}, Created: {datetime.fromtimestamp(ctime).strftime('%Y-%m-%d %H:%M:%S')}\n")
    print_if_not_quiet("Wrote sorted PDF list by creation date to 'pdf_list.txt'")
    
    return [(pdf, size, pages) for pdf, size, pages, _ in pdf_data], total_per_code, folder_path

def analyze_pdfs(pdf_data, folder, dpi, db_name, force, workers, extract_text=False, length=None, quick=False, ocr=False):
    """Analyze PDFs with same pages and size using multiple threads."""
    print_if_not_quiet("Starting analysis of PDFs for duplicates based on pages and size")
    from collections import defaultdict
    groups = defaultdict(list)
    for pdf, size, pages in pdf_data:
        groups[(pages, size)].append(pdf)
    
    total_groups = sum(1 for pdfs in groups.values() if len(pdfs) > 1)
    total_to_process = sum(len(pdfs) for pdfs in groups.values() if len(pdfs) > 1)
    print_if_not_quiet(f"Identified {total_groups} groups of PDFs with identical page counts and sizes, total to process: {total_to_process}")
    
    if total_to_process == 0:
        print_if_not_quiet("No duplicate PDFs found to analyze")
        return []
    
    skipped = []
    processed_count = 0
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_pdf = {}
        for (pages, size), pdfs in groups.items():
            if len(pdfs) > 1:
                print_if_not_quiet(f"Queueing group: {len(pdfs)} PDFs with {pages} pages, {format_size(size)}")
                for pdf in pdfs:
                    future_to_pdf[executor.submit(process_pdf, os.path.join(folder, pdf), dpi, db_name, extract_text, force, length=length, quick=quick, ocr=ocr, workers=workers)] = pdf
        
        for i, future in enumerate(as_completed(future_to_pdf), 1):
            pdf = future_to_pdf[future]
            try:
                doc_numbers, full_text, pgs, sz = future.result()
                if doc_numbers:
                    processed_count += 1
                    print_if_not_quiet(f"Progress: {i}/{total_to_process} - {pdf}: Processed successfully (Total processed: {processed_count})")
                else:
                    skipped.append(pdf)
                    print_if_not_quiet(f"Progress: {i}/{total_to_process} - {pdf}: Skipped (no doc number or already processed)")
            except Exception as e:
                skipped.append(pdf)
                print_if_not_quiet(f"Progress: {i}/{total_to_process} - {pdf}: Failed with error: {str(e)}")
    
    print_if_not_quiet(f"Analysis completed - Processed: {processed_count}, Skipped: {len(skipped)}")
    return skipped

def main():
    parser = argparse.ArgumentParser(description="wookey-extract-docnr.py - PDF Document Number Extractor")
    parser.add_argument("--folder", required=True, help="Folder containing PDF files, a single PDF file, or a wildcard pattern (e.g., 'path/*.pdf')")
    parser.add_argument("--db", required=True, help="Database file name to store results")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of all PDFs")
    parser.add_argument("--text", action="store_true", help="Extract and save full text of pages")
    parser.add_argument("--quiet", action="store_true", help="Suppress all output messages")
    parser.add_argument("--verbose", action="store_true", help="Verbose messages")
    parser.add_argument("--ocr", action="store_true", help="Forces OCR on pages")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads (default: 4)")
    parser.add_argument("--length", type=int, default=None, help="Exact length of the digit sequence in document numbers to match (e.g., 4 for exactly 4 digits)")
    parser.add_argument("--quick", action="store_true", help="Quick mode: only check first and last lines, prefer structure matching previous number")
    parser.add_argument("--dpi", type=int, default=100, help="DPI mode to use, e.g. 300")
    args = parser.parse_args()
    
    global quiet_mode
    global VERBOSE_MODE
  
    quiet_mode = args.quiet
    print_if_not_quiet(f"Script started - Quiet mode: {quiet_mode}, Folder/PDF: {args.folder}, DB: {args.db}, Workers: {args.workers}, Length: {args.length}, Quick: {args.quick}, Ocr: {args.ocr}, Verbose: {args.verbose}, Dpi: {args.dpi}")
    
    folder = args.folder
    db_name = args.db
    force = args.force
    dpi = args.dpi
    workers = max(1, args.workers)
    length = args.length
    quick = args.quick
    ocr = args.ocr
    VERBOSE_MODE = args.verbose
    
    print_if_not_quiet(f"Establishing initial database connection to: {db_name}")
    db_conn = sqlite3.connect(db_name)
    
    print_if_not_quiet("Scanning folder or processing single PDF...")
    pdf_data, total_per_code, folder_path = scan_folder_and_list(folder, db_conn, force=force)
    total_pages = sum(p[2] for p in pdf_data)
    total_size = sum(p[1] for p in pdf_data)
    print_if_not_quiet(f"\nPDF Files Found: {len(pdf_data)} (sorted by creation date ascending)")
    print_if_not_quiet(f"Total Pages: {total_pages}, Total Size: {format_size(total_size)}")
    for i, (pdf, size, pages) in enumerate(pdf_data, 1):
        base_folder = folder_path
        ctime = os.path.getctime(os.path.join(base_folder, pdf))
        print_if_not_quiet(f"{i}. {pdf} - {pages} pages, {format_size(size)}, Created: {datetime.fromtimestamp(ctime).strftime('%Y-%m-%d %H:%M:%S')}")
    print_if_not_quiet(f"\nTotal per 32-char code: {total_per_code}")
    print_if_not_quiet(f"Unique codes detected: {len(total_per_code)}")
    
    skipped = []
    pages_handled = 0
    size_handled = 0
    start_time = time.time()
    
    while True:
        choice = input("Enter choice (a=analyze PDF, b=analyze duplicates, c=extract document numbers, q=quit): ").lower()
        if choice == 'q':
            break
        elif choice == 'a':
            num = int(input("Enter PDF number: ")) - 1 if not quiet_mode else 0
            if 0 <= num < len(pdf_data):
                pdf, size, pages = pdf_data[num]
                base_folder = folder_path
                pdf_path = os.path.join(base_folder, pdf)
                ctime = os.path.getctime(pdf_path)
                print_if_not_quiet(f"Selected PDF #{num + 1}: {pdf} ({pages} pages, {format_size(size)}, Created: {datetime.fromtimestamp(ctime).strftime('%Y-%m-%d %H:%M:%S')})")
                
                # Prompt for page range
                page_range_input = input("Page range? (e.g., 1,3,5,7-13,318,479-512, or 'all' for all pages): ").strip()
                if page_range_input.lower() == 'all':
                    page_range_input = f"1-{pages}"
                
                # Parse page range
                page_set = set()
                try:
                    splitted = False
                    for part in page_range_input.split(','):
                        splitted = True
                        part = part.strip()
                        if '-' in part:
                            start, end = map(int, part.split('-'))
                            page_set.update(range(start, end + 1))
                        else:
                            page_set.add(int(part))
                    # Validate page numbers
                    if not splitted:
                        if '-' in page_range_input:
                            print('Received input', splitted, page_range_input)
                            start, end = map(int, page_range_input.split('-'))
                            page_set.update(range(start, end + 1))
                        else:
                            page_set.add(int(part))
                    pages_to_process = sorted(list(page_set))
                    # invalid_pages = [p for p in page_set if p < 1 or p > pages]
                    # if invalid_pages:
                        # print(pages_to_process)
                        # raise ValueError(f"Invalid page numbers: {invalid_pages}")
                    # pages_to_process = sorted(list(page_set))
                    print_if_not_quiet(f"Processing pages: {pages_to_process}")
                except ValueError as e:
                    print_if_not_quiet(f"Error: {e}. Please use format like '1,3,5,7-13'. Skipping this PDF.")
                    continue
                
                # Process only the specified page range
                doc_numbers, full_text, pgs, sz = process_pdf(pdf_path, dpi, db_name, extract_text=args.text, 
                                                            force=True if os.path.isfile(folder) else force, 
                                                            length=length, quick=quick, ocr=ocr, workers=workers,
                                                            page_range=pages_to_process)
                if doc_numbers:
                    pages_handled += pgs
                    size_handled += sz
                    print_if_not_quiet(f"Completed processing '{pdf}' - Pages handled: {pgs}, Size: {format_size(sz)}")
                else:
                    skipped.append(pdf)
                    print_if_not_quiet(f"Skipped '{pdf}' due to no document number or prior processing")
            else:
                print_if_not_quiet(f"Invalid PDF number. Please enter a number between 1 and {len(pdf_data)}.")
        
        elif choice == 'b':
            skipped = analyze_pdfs(pdf_data, folder if not os.path.isfile(folder) else os.path.dirname(folder), dpi, db_name, force, workers, args.text, length=length, quick=quick, ocr=ocr)
            for pdf, size, pages in pdf_data:
                if pdf not in skipped:
                    pages_handled += pages
                    size_handled += size
            print_if_not_quiet(f"Option b completed - Pages handled: {pages_handled}/{total_pages}, Size: {format_size(size_handled)}/{format_size(total_size)}")
        
        elif choice == 'c':
            print_if_not_quiet(f"Processing all {len(pdf_data)} PDFs in creation date order with {workers} workers")
            base_folder = folder_path
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_pdf = {
                    executor.submit(process_pdf, os.path.join(base_folder, pdf), dpi, db_name, args.text, True if os.path.isfile(folder) else force, length=length, quick=quick, ocr=ocr, workers=workers, page_range=None): pdf
                    for pdf, size, pages in pdf_data
                }
                for i, future in enumerate(as_completed(future_to_pdf), 1):
                    pdf = future_to_pdf[future]
                    try:
                        doc_numbers, full_text, pgs, sz = future.result()
                        if doc_numbers:
                            pages_handled += pgs
                            size_handled += sz
                            print_if_not_quiet(f"Progress: {i}/{len(pdf_data)} - Processed '{pdf}', Pages handled: {pages_handled}/{total_pages}")
                        else:
                            skipped.append(pdf)
                            print_if_not_quiet(f"Progress: {i}/{len(pdf_data)} - Skipped '{pdf}'")
                    except Exception as e:
                        print_if_not_quiet(f"Progress: {i}/{len(pdf_data)} - Failed '{pdf}' with error: {str(e)}")
                        skipped.append(pdf)
        
        if skipped:
            with open('wookey-extract-docnr-skipped.list', 'a') as f:
                for pdf in skipped:
                    f.write(f"{pdf}\n")
            print_if_not_quiet(f"Skipped {len(skipped)} PDFs: {', '.join(skipped)}")
            print_if_not_quiet("Updated 'wookey-extract-docnr-skipped.list' with skipped files")
        
        pages_left = total_pages - pages_handled
        size_left = total_size - size_handled
        elapsed = time.time() - start_time
        if pages_handled >= 400 or choice in ['a', 'b'] or pages_handled == total_pages:
            rate = pages_handled / elapsed if elapsed > 0 else 0
            time_left = pages_left / rate if rate > 0 else 0
            print_if_not_quiet(f"Progress Report - Pages: {pages_handled}/{total_pages} | Left: {pages_left} | "
                               f"Size: {format_size(size_handled)}/{format_size(total_size)} | "
                               f"Left: {format_size(size_left)} | Time Elapsed: {format_time(elapsed)} | "
                               f"Time Left: {format_time(time_left)}")
        print_if_not_quiet(f"Operation '{choice}' completed in {format_time(elapsed)}")
    
    print_if_not_quiet("Closing initial database connection")
    db_conn.close()
    print_if_not_quiet("Script execution finished")
    
if __name__ == "__main__":
    main()

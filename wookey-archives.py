import os
import zipfile
import sqlite3
from datetime import datetime
import argparse
import shutil
import re
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import Counter
import PyPDF2
import pytesseract
from pdf2image import convert_from_path
import subprocess

# Global SQLite connection to avoid repeated connections
conn = sqlite3.connect(':memory:', check_same_thread=False)
cursor = conn.cursor()

def setup_db(db_path, add_text=False):
    global conn, cursor
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS documents_archives (
        Id INTEGER PRIMARY KEY AUTOINCREMENT,
        fromFile TEXT,
        fromFilePath TEXT,
        documentName TEXT,
        documentType TEXT,
        documentCompressedSize INTEGER,
        documentSize INTEGER,
        documentCompressRatio REAL,
        documentDateModified TEXT,
        dossierReference TEXT,
        dossierReferenceAlias TEXT,
        createdDate TEXT,
        lastSeen TEXT
    )
    ''')
    cursor.execute('PRAGMA table_info(documents_archives)')
    columns = {col[1].lower() for col in cursor.fetchall()}
    if 'dossierreference' not in columns:
        cursor.execute('ALTER TABLE documents_archives ADD COLUMN dossierReference TEXT')
    if 'dossierreferencealias' not in columns:
        cursor.execute('ALTER TABLE documents_archives ADD COLUMN dossierReferenceAlias TEXT')
    if 'createddate' not in columns:
        cursor.execute('ALTER TABLE documents_archives ADD COLUMN createdDate TEXT')
    if 'lastseen' not in columns:
        cursor.execute('ALTER TABLE documents_archives ADD COLUMN lastSeen TEXT')
    if add_text:
        if 'text' not in columns:
            cursor.execute('ALTER TABLE documents_archives ADD COLUMN text TEXT')
        if 'notextdate' not in columns:
            cursor.execute('ALTER TABLE documents_archives ADD COLUMN noTextDate TEXT')
        if 'ocrtext' not in columns:
            cursor.execute('ALTER TABLE documents_archives ADD COLUMN ocrText TEXT')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_text ON documents_archives (text)')
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fromFile ON documents_archives (fromFile)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_documentName ON documents_archives (documentName)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_dossierReference ON documents_archives (dossierReference)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_lastSeen ON documents_archives (lastSeen)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_createdDate ON documents_archives (createdDate)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ocrText ON documents_archives (ocrText)")
    conn.commit()

def extract_text_from_pdf(pdf_path, zip_idx, total_zips, doc_idx, total_docs):
    prefix = f"({zip_idx}/{total_zips}) ({doc_idx}/{total_docs})"
    print(f"{prefix} Extracting text from {pdf_path}")
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
            if text.strip():
                print(f"{prefix} Text extracted successfully from {pdf_path} (length: {len(text)})")
                return text.strip()
            else:
                print(f"{prefix} No text found in {pdf_path}")
                return None
    except Exception as e:
        print(f"{prefix} Error extracting text from {pdf_path}: {e}")
        return None

def check_poppler():
    try:
        subprocess.run(['pdftoppm', '-v'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Poppler not installed or not in PATH. OCR will be skipped.")
        return False

def ocr_pdf(pdf_path, zip_idx, total_zips, doc_idx, total_docs, doc_name):
    prefix = f"({zip_idx}/{total_zips}) ({doc_idx}/{total_docs})"
    print(f"{prefix} Attempting OCR on {pdf_path}")
    tabs = '\t'
    if not check_poppler():
        return None
    try:
        images = convert_from_path(pdf_path)
        ocr_text = ""
        for i, image in enumerate(images):
            prefix = f"{tabs}({zip_idx}/{total_zips}) ({doc_idx}/{total_docs}) (OCR Page {i+1})"
            print(f"{prefix} {doc_name}")
            text = pytesseract.image_to_string(image, lang='nld')
            if text.strip():
                ocr_text += text
            else:
                print(f"{prefix} {doc_name} (retrying with English)")
                text = pytesseract.image_to_string(image, lang='eng')
                ocr_text += text
                
        if ocr_text.strip():
            print(f"{prefix} OCR completed, length: {len(ocr_text)}")
            return ocr_text.strip()
        else:
            print(f"{prefix} No text extracted via OCR")
            return None
    except Exception as e:
        print(f"--> {prefix} Error OCRing {pdf_path}: {e}")
        return None

def truncate_filename(filename, temp_dir, max_total_length=260, no_duplicates=False):
    max_filename_length = max_total_length - len(temp_dir) - 1
    if max_filename_length < 50:
        raise ValueError(f"Temporary directory path '{temp_dir}' is too long.")
    if len(filename) <= max_filename_length:
        return filename
    if no_duplicates:
        print(f"Skipping rename for {filename} due to --noduplicates; file will not be processed if it already exists.")
        return None  # Indicates to skip this file
    name, ext = os.path.splitext(filename)
    base_length = max_filename_length - len(ext) - 5
    truncated_name = name[:base_length] + f"_{hash(filename) % 10000:04d}"
    truncated_filename = truncated_name + ext
    print(f"Truncating filename from {len(filename)} to {len(truncated_filename)} for path '{temp_dir}': {filename} -> {truncated_filename}")
    return truncated_filename

def process_pdf(args):
    row, temp_dir, zip_idx, total_zips, doc_idx, total_docs, no_tmp_clean = args
    id_, doc_name = row
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    prefix = f"({zip_idx}/{total_zips}) ({doc_idx}/{total_docs})"
    
    if not doc_name.lower().endswith('.pdf'):
        print(f"{prefix} Skipping {doc_name} - not a PDF")
        return (id_, None, None, None, "Not a PDF")

    truncated_doc_name = truncate_filename(doc_name, temp_dir)
    if truncated_doc_name is None:  # Skip if truncate_filename returns None (due to --noduplicates)
        return (id_, None, None, None, "Skipped due to duplicate with --noduplicates")
    pdf_path = os.path.join(temp_dir, truncated_doc_name)
    
    if len(pdf_path) > 260:
        print(f"{prefix} Warning: Path '{pdf_path}' still exceeds 260 characters after truncation!")
    
    print(f"{prefix} Processing PDF: {doc_name} (using {truncated_doc_name})")
    
    try:
        text = extract_text_from_pdf(pdf_path, zip_idx, total_zips, doc_idx, total_docs)
        if text:
            return (id_, text, None, None, "Text extracted")
        else:
            ocr_text = ocr_pdf(pdf_path, zip_idx, total_zips, doc_idx, total_docs, doc_name)
            if ocr_text:
                return (id_, None, now, ocr_text, "OCR successful")
            else:
                return (id_, None, now, None, "No text from OCR")
    except Exception as e:
        print(f"{prefix} Error processing {doc_name}: {e}")
        return (id_, None, now, None, f"Processing error: {e}")
    finally:
        if not no_tmp_clean and os.path.exists(pdf_path):
            print(f"{prefix} Cleaning up {pdf_path}")
            os.remove(pdf_path)
        elif no_tmp_clean and os.path.exists(pdf_path):
            print(f"{prefix} Keeping temporary file {pdf_path} as requested (--notmpclean)")
        else:
            print(f"{prefix} No cleanup needed for {pdf_path} - file not created")

def extract_with_custom_names(zip_ref, members, temp_dir, total_files, max_extraction_workers=4, no_duplicates=False):
    """Extract files from a ZIP with custom output filenames, parallelized."""
    def extract_single_file(args):
        member, file_idx = args
        original_name = member
        truncated_name = truncate_filename(original_name, temp_dir, no_duplicates=no_duplicates)
        if truncated_name is None:  # Skip if --noduplicates and file would be renamed
            return
        output_path = os.path.join(temp_dir, truncated_name)
        if len(output_path) > 260:
            raise ValueError(f"Output path '{output_path}' exceeds 260 characters.")
        prefix = f"({file_idx}/{total_files})"
        print(f"{prefix} Extracting '{original_name}' as '{truncated_name}' to '{temp_dir}'")
        with zip_ref.open(original_name) as source, open(output_path, 'wb') as target:
            shutil.copyfileobj(source, target)

    with ThreadPoolExecutor(max_workers=max_extraction_workers) as executor:
        executor.map(extract_single_file, [(member, i) for i, member in enumerate(members, 1)])

def process_text_addition(db_path, max_workers=8, no_tmp_clean=False, no_duplicates=False):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT DISTINCT fromFilePath 
        FROM documents_archives 
        WHERE documentType = "pdf" 
        AND fromFilePath IN (
            SELECT fromFilePath 
            FROM documents_archives 
            WHERE documentType = "pdf" 
            AND (text IS NULL AND ocrText IS NULL)
            GROUP BY fromFilePath
        )
    ''')
    zip_files = [row[0] for row in cursor.fetchall()]
    total_zips = len(zip_files)
    
    cursor.execute('SELECT COUNT(*) FROM documents_archives WHERE documentType = "pdf" AND (text IS NULL AND ocrText IS NULL)')
    total_pdfs = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM documents_archives WHERE documentType = "pdf" AND (text IS NOT NULL OR ocrText IS NOT NULL)')
    skipped_pdfs = cursor.fetchone()[0]
    
    print(f"(0/{total_zips}) Starting text extraction for {total_zips} zip files with partial/no text extracted, containing {total_pdfs} PDFs needing processing (skipping {skipped_pdfs} already processed)...")
    
    conn.close()
    
    temp_base_dir = './tmp'
    os.makedirs(temp_base_dir, exist_ok=True)
    
    processed_pdfs = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for zip_idx, zip_path in enumerate(zip_files, 1):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            temp_dir = os.path.join(temp_base_dir, f"z{zip_idx}")
            os.makedirs(temp_dir, exist_ok=True)
            
            print(f"({zip_idx}/{total_zips}) Processing zip: {zip_path}")
            try:
                cursor.execute('''
                    SELECT Id, documentName 
                    FROM documents_archives 
                    WHERE fromFilePath = ? 
                    AND documentType = "pdf" 
                    AND (text IS NULL AND ocrText IS NULL)
                    AND documentName NOT LIKE '%Dataset_Deel_%'
                    AND documentName NOT IN ('VWS-WOO-06b-866195-2021_03_29_Analyse-_en_brondocument.xlsx.pdf', 
                                             'VWS-WOO-06b-1644428-2021_03_29_COVID-19_nieuw_besmette_locaties.pdf', 
                                             'VWS-WOO-08b-1410019-surveyresults_korteinterviews_r8_Deel_1.pdf', 
                                             'VWS-WOO-08b-1410018-surveyresults_korteinterviews_r8_Deel_2.pdf',
                                             'VWS-WOO-04-557627-COVID-19-geographic-disbtribution-worldwide-2020-09-28.csv.pdf',
                                             'VWS-WOO-04-557630-COVID-19-geographic-disbtribution-worldwide-2020-09-30.csv.pdf',
                                             'VWS-WOO-08-1293291-Metadata_testbeleid_gedrag_bereidheid__ARK_Deel_2.pdf', 
                                             'VWS-WOO-08-1293290-Metadata_testbeleid_gedrag_bereidheid__ARK_Deel_1.pdf',
                                             'VWS-WOO-08-1155097-The_Vaccine_Adverse_Event_Reporting_System__VAERS__Results_Form.pdf')
                ''', (zip_path,))
                zip_pdfs = cursor.fetchall()
                
                if not zip_pdfs:
                    print(f"({zip_idx}/{total_zips}) No PDFs need processing in {zip_path} - all already have text.")
                    continue
                
                total_docs = len(zip_pdfs)
                print(f"({zip_idx}/{total_zips}) Extracting {total_docs} PDFs needing text from {zip_path} to {temp_dir}")
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    pdf_files_to_extract = [row[1] for row in zip_pdfs if row[1] in zip_ref.namelist()]
                    if not pdf_files_to_extract:
                        print(f"({zip_idx}/{total_zips}) No matching PDFs found in {zip_path} for extraction.")
                        continue
                    extract_with_custom_names(zip_ref, pdf_files_to_extract, temp_dir, total_docs, no_duplicates=no_duplicates)
                
                pdf_args = [(row, temp_dir, zip_idx, total_zips, doc_idx, total_docs, no_tmp_clean) 
                            for doc_idx, row in enumerate(zip_pdfs, 1)]
                results = list(executor.map(process_pdf, pdf_args))
                
                print(f"({zip_idx}/{total_zips}) Batching updates for {len(results)} PDFs from {zip_path}")
                try:
                    cursor.execute('BEGIN TRANSACTION')
                    for id_, text, no_text_date, ocr_text, status in results:
                        cursor.execute('''
                            UPDATE documents_archives 
                            SET text = ?, noTextDate = ?, ocrText = ? 
                            WHERE Id = ?
                        ''', (text, no_text_date, ocr_text, id_))
                        processed_pdfs += 1
                        print(f"({zip_idx}/{total_zips}) Updated ID {id_}: {status}")
                    cursor.execute('COMMIT')
                    print(f"({zip_idx}/{total_zips}) Database commit successful for {zip_path}. Processed {processed_pdfs}/{total_pdfs} PDFs so far.")
                except Exception as e:
                    print(f"({zip_idx}/{total_zips}) Error committing transaction for {zip_path}: {e}")
                    cursor.execute('ROLLBACK')
            
            except zipfile.BadZipFile:
                print(f"({zip_idx}/{total_zips}) Bad zip file: {zip_path}. Skipping.")
            except Exception as e:
                print(f"({zip_idx}/{total_zips}) Error processing zip {zip_path}: {e}")
            finally:
                if not no_tmp_clean:
                    print(f"({zip_idx}/{total_zips}) Cleaning up temporary directory {temp_dir}")
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                else:
                    print(f"({zip_idx}/{total_zips}) Keeping temporary directory {temp_dir} as requested (--notmpclean)")
                conn.commit()
                conn.close()
    
    if os.path.exists(temp_base_dir):
        if not os.listdir(temp_base_dir):  # Check if directory is empty
            print(f"({total_zips}/{total_zips}) Removing empty temporary base directory {temp_base_dir}")
            shutil.rmtree(temp_base_dir)
        elif not no_tmp_clean:
            print(f"({total_zips}/{total_zips}) Final cleanup: removing {temp_base_dir}")
            shutil.rmtree(temp_base_dir)
        else:
            print(f"({total_zips}/{total_zips}) Keeping temporary base directory {temp_base_dir} as requested (--notmpclean)")
    else:
        print(f"({total_zips}/{total_zips}) No temporary base directory {temp_base_dir} to clean up")

    print(f"({total_zips}/{total_zips}) Text extraction and OCR process completed. Processed {processed_pdfs}/{total_pdfs} PDFs needing processing (skipped {skipped_pdfs} already processed) across {total_zips} zip files.")

def get_file_cache():
    cursor.execute("SELECT fromFile, documentName FROM documents_archives")
    return {f"{row[0]}{row[1]}": row for row in cursor.fetchall()}

def process_zip_file(zip_path, file_cache, file_counter, total_size):
    file = os.path.basename(zip_path)
    match = re.search(r'besluit-(\d+)', file, re.IGNORECASE)
    dossier_reference = match.group(1) if match else None
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    actions = []
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_size = os.path.getsize(zip_path)
            num_files = len(zip_ref.infolist())
            
            print(f"Processing file {file_counter} of {len(total_size)}: {file}, Size: {file_size} bytes, Embedded Files: {num_files}")
            for zip_info in zip_ref.infolist():
                if zip_info.file_size > 0:
                    compress_ratio = (zip_info.compress_size / zip_info.file_size) * 100
                else:
                    compress_ratio = 0
                
                key = f"{file}{zip_info.filename}"
                if key in file_cache:
                    actions.append(('UPDATE', file_cache[key][0], {'lastSeen': now}))
                else:
                    actions.append(('INSERT', None, {
                        'fromFile': file, 
                        'fromFilePath': zip_path, 
                        'documentName': zip_info.filename, 
                        'documentType': os.path.splitext(zip_info.filename)[1][1:],
                        'documentCompressedSize': zip_info.compress_size, 
                        'documentSize': zip_info.file_size, 
                        'documentCompressRatio': compress_ratio,
                        'documentDateModified': datetime(*zip_info.date_time).strftime('%Y-%m-%d %H:%M:%S'),
                        'dossierReference': dossier_reference, 
                        'dossierReferenceAlias': None, 
                        'createdDate': now, 
                        'lastSeen': now
                    }))
    except zipfile.BadZipFile:
        print(f"Error: {file} is not a valid zip file. Renaming.")
        new_name = os.path.splitext(file)[0] + '_WITH_ERROR' + os.path.splitext(file)[1]
        new_path = os.path.join(os.path.dirname(zip_path), new_name)
        shutil.move(zip_path, new_path)
    except Exception as e:
        print(f"An error occurred with {file}: {e}")
        new_name = os.path.splitext(file)[0] + '_WITH_ERROR' + os.path.splitext(file)[1]
        new_path = os.path.join(os.path.dirname(zip_path), new_name)
        shutil.move(zip_path, new_path)
    
    return actions

def execute_batch(actions):
    for action, id, data in actions:
        if action == 'UPDATE':
            cursor.execute('''
            UPDATE documents_archives 
            SET lastSeen = ?
            WHERE Id = ?
            ''', (data['lastSeen'], id))
        elif action == 'INSERT':
            cursor.execute('''
            INSERT INTO documents_archives (
                fromFile, fromFilePath, documentName, documentType, 
                documentCompressedSize, documentSize, documentCompressRatio, 
                documentDateModified, dossierReference, dossierReferenceAlias,
                createdDate, lastSeen
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', tuple(data.values()))

def check_duplicates():
    cursor.execute("SELECT COUNT(DISTINCT fromFile) FROM documents_archives")
    archive_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM documents_archives")
    total_files = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM (SELECT documentName FROM documents_archives GROUP BY documentName HAVING COUNT(*) > 1)')
    duplicates_by_name = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM (SELECT documentName, documentSize FROM documents_archives GROUP BY documentName, documentSize HAVING COUNT(*) > 1)')
    duplicates_by_name_size = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM (SELECT documentName, documentDateModified FROM documents_archives GROUP BY documentName, documentDateModified HAVING COUNT(*) > 1)')
    duplicates_by_name_date = cursor.fetchone()[0]

    print(f"1. Number of different archive files: {archive_count}")
    print(f"2. Total number of files: {total_files}")
    print(f"3. Files duplicate by name: {duplicates_by_name}")
    print(f"4. Files duplicate by name and size: {duplicates_by_name_size}")
    print(f"5. Files duplicate by name and date modified: {duplicates_by_name_date}")

    cursor.execute('SELECT documentName, COUNT(*) AS count FROM documents_archives GROUP BY documentName')
    file_occurrences = cursor.fetchall()
    occurrence_counts = Counter(count for _, count in file_occurrences)
    top_occurrences = sorted(occurrence_counts.items(), key=lambda x: x[0], reverse=True)[:5]

    print("\nTotal number of files found X times (top 5):")
    for occurrence, count in top_occurrences:
        print(f"  {count} files found {occurrence} time(s)")

    cursor.execute('SELECT documentName FROM documents_archives')
    archive_names = set(row[0] for row in cursor.fetchall())
    cursor.execute('SELECT [Document naam] FROM documents')
    doc_names = set(row[0] for row in cursor.fetchall())

    only_in_archives = archive_names - doc_names
    print(f"\nNumber of files in documentNaam but not in [Document naam]: {len(only_in_archives)}")
    only_in_documents = doc_names - archive_names
    print(f"Number of files in [Document naam] but not in documentNaam: {len(only_in_documents)}")
    
    cursor.execute('SELECT documentName, fromFile, documentSize, documentDateModified FROM documents_archives GROUP BY documentName, fromFile ORDER BY documentName')
    all_files = {name: [] for name, _, _, _ in cursor.fetchall()}
    cursor.execute('SELECT documentName, fromFile, documentSize, documentDateModified FROM documents_archives ORDER BY documentName')
    for name, from_file, size, date_modified in cursor.fetchall():
        all_files[name].append((from_file, size, date_modified))

    top_files = sorted(all_files.items(), key=lambda x: len(x[1]), reverse=True)[:25]
    print("\nTop 25 filenames found the most:")
    for name, occurrences in top_files:
        print(f"  {name}: {len(occurrences)} occurrences")
        sorted_occurrences = sorted(occurrences, key=lambda x: x[2])
        for from_file, size, date_modified in sorted_occurrences:
            dt = datetime.strptime(date_modified, '%Y-%m-%d %H:%M:%S')
            print(f"\t{from_file} - Size: {size} bytes - Created: {dt.strftime('%Y-%m-%d %H:%M:%S')}")

    cursor.execute('SELECT documentName, COUNT(*) FROM (SELECT documentName, documentSize FROM documents_archives GROUP BY documentName, documentSize) GROUP BY documentName HAVING COUNT(*) > 1')
    same_name_different_size = cursor.fetchall()
    print("\nFiles with same name but different size:")
    count_different_size = sum(count for _, count in same_name_different_size)
    print(f"  Total: {count_different_size}")
    for name, count in same_name_different_size:
        print(f"  - {name}: {count} different sizes")

def process_folder(folder_path, empty, max_workers=8):
    if empty:
        cursor.execute('DELETE FROM documents_archives')
        conn.commit()
        print("Table 'documents_archives' has been emptied.")

    file_cache = get_file_cache()
    
    zip_files = [os.path.join(root, f) for root, _, files in os.walk(folder_path) for f in files if f.endswith('.zip')]
    total_size = [(f, os.path.getsize(f)) for f in zip_files]
    
    print(f"Found {len(zip_files)} zip files. Total size: {sum(size for _, size in total_size)} bytes.")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        all_actions = []
        file_counter = 1
        for actions in executor.map(lambda f: process_zip_file(f, file_cache, file_counter, total_size), zip_files):
            file_counter += 1
            all_actions.extend(actions)
            for action, _, data in actions:
                if action == 'INSERT':
                    key = f"{data['fromFile']}{data['documentName']}"
                    file_cache[key] = (None, )
        cursor.execute('BEGIN TRANSACTION')
        execute_batch(all_actions)
        cursor.execute('COMMIT')
    
    conn.commit()

def query_and_extract(db_path, query_words):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    conditions = " AND ".join(["documentName LIKE ?" for _ in query_words])
    params = [f"%{word}%" for word in query_words]
    cursor.execute(f'SELECT fromFilePath, documentName FROM documents_archives WHERE {conditions}', params)
    matches = cursor.fetchall()

    if not matches:
        print("No matches found for the given query words.")
        conn.close()
        return

    print(f"Found {len(matches)} matches:")
    for i, (from_file_path, doc_name) in enumerate(matches, 1):
        print(f"{i}. {doc_name} from {from_file_path}")

    response = input("\nDo you want to extract these files? (yes/no): ").strip().lower()
    if response == 'yes':
        extract_folder = './extractions'
        os.makedirs(extract_folder, exist_ok=True)
        for from_file_path, doc_name in matches:
            try:
                with zipfile.ZipFile(from_file_path, 'r') as zip_ref:
                    base_name, ext = os.path.splitext(doc_name)
                    target_path = os.path.join(extract_folder, doc_name)
                    counter = 1
                    while os.path.exists(target_path):
                        target_path = os.path.join(extract_folder, f"{base_name}_{counter}{ext}")
                        counter += 1
                    zip_ref.extract(doc_name, extract_folder)
                    if target_path != os.path.join(extract_folder, doc_name):
                        shutil.move(os.path.join(extract_folder, doc_name), target_path)
                    print(f"Extracted: {target_path}")
            except Exception as e:
                print(f"Failed to extract {doc_name} from {from_file_path}: {e}")
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process zip files and store metadata in SQLite.")
    parser.add_argument("--db", required=True, help="Database filename")
    parser.add_argument("--empty", action="store_true", help="Empty the table before processing")
    parser.add_argument("--check", action="store_true", help="Check for duplicate filenames")
    parser.add_argument("--query", nargs='+', help="Search and extract files by documentName")
    parser.add_argument("--text", action="store_true", help="Add and populate text fields for PDFs with OCR")
    parser.add_argument("--notmpclean", action="store_true", help="Prevent removal of temporary files and directories")
    parser.add_argument("--noduplicates", action="store_true", help="Skip renaming and processing of duplicate filenames")
    parser.add_argument("folder_path", nargs='?', default=None, help="Path to folder with zip files")
    args = parser.parse_args()

    setup_db(args.db, add_text=args.text)

    if args.check:
        check_duplicates()
    elif args.query:
        query_and_extract(args.db, args.query)
    elif args.text:
        process_text_addition(args.db, max_workers=8, no_tmp_clean=args.notmpclean, no_duplicates=args.noduplicates)
    elif args.folder_path:
        process_folder(args.folder_path, args.empty)
    else:
        print("Please provide a folder path, or use --check, --query, or --text.")



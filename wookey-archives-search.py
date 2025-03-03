import argparse
import sqlite3
import re
import sys
import os
import shutil
import zipfile
from typing import List, Dict, Tuple
import pandas as pd
from datetime import datetime
from collections import Counter
import time
import json

DEBUG_MODE = False
VERBOSE_MODE = False

def parse_search_terms(search_string: str) -> tuple[List[str], List[str]]:
    """Parse search terms into include and exclude lists."""
    includes = []
    excludes = []
    if not search_string:
        return includes, excludes
        
    terms = search_string.split(',')
    for term in terms:
        term = term.strip()
        if term.startswith('-'):
            excludes.append(term[1:])
        else:
            includes.append(term)
    return includes, excludes

def split_document_name(doc_name: str) -> Dict[str, str]:
    """Split documentName into components."""
    parts = doc_name.split('-', 5)
    if len(parts) < 6:
        return {'indicatorRequestNumber': '', 'indicatorDocumentID': ''}
    
    return {
        'indicatorMinistry': parts[0],
        'indicatorRequest': parts[1],
        'indicatorRequestNumber': parts[2],
        'indicatorDocumentID': parts[3],
        'indicatorDocumentNameOrSubject': parts[4],
        'indicatorFileType': parts[5]
    }

def lookup_in_documents(conn: sqlite3.Connection, request_number: str, document_id: str) -> Tuple[str, str, str, str]:
    """Lookup document in documents and get productionDate and Title from documents_tags."""
    cursor = conn.cursor()
    query = "SELECT d.ModifiedDate, d.[Locatie open.minvws.nl], coalesce(dt.tag_value,'--') AS publicationDate, coalesce(dt2.tag_value,'--') AS documentName FROM documents d LEFT OUTER JOIN documents_tags dt ON d.[Locatie open.minvws.nl] = dt.Title AND dt.tag_key = 'document_documentPublicationDate' LEFT OUTER JOIN documents_tags dt2 ON d.[Locatie open.minvws.nl] = dt2.Title AND dt2.tag_key = 'document_documentName' WHERE d.[Locatie open.minvws.nl] LIKE ?"
    param = f"%{request_number}-{document_id}%"
    cursor.execute(query, (param,))
    result = cursor.fetchone()
    modified_date = result[0] if result and result[0] else "N/A"
    locatie = result[1] if result and result[1] else "N/A"
    production_date = result[2] if result and result[2] else "N/A"
    production_date = fix_date(production_date)
    title = result[3] if result and result[3] else "N/A"
    
    return (modified_date, production_date, locatie, title)

# Regex requires '@' and matches only user@host.tld or partials with '@'
EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9._%+-]*@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    
def fix_date(date_str: str) -> str:
    """
    Convert a date string from incorrect format (e.g., '0001-10-20') to correct format (e.g., '2020-10-01').
    Assumes the year is truncated, and day/month are swapped with incorrect day values.
    
    Args:
        date_str (str): Input date in format 'YYYY-MM-DD' (e.g., '0001-10-20')
    
    Returns:
        str: Corrected date in format 'YYYY-MM-DD' (e.g., '2020-10-01')
    """
    try:
        # Split the date into components
        year, month, day = date_str.split('-')
        
        if year.startswith('00'):
            correct_year = '20' + day + '-' + month + '-' + year.replace('00','')
        else:
            correct_year = date_str
                    
        # Format the corrected date
        return f"{correct_year}"
    except (ValueError, IndexError):
        # Return original string if parsing fails
        return date_str

# Global counters for cumulative totals
TOTAL_USERS = set()
TOTAL_HOSTS = set()
TOTAL_COMPLETE = 0

def get_context_words(words: list, char_positions: list, match_start: int, match_end: int, context_size: int) -> str:
    """Extract context words around a match using pre-split words and positions."""
    if DEBUG_MODE:
        print(f"  Extracting context for match at {match_start}-{match_end}")
    
    start_idx = end_idx = -1
    for i, (word, w_start, w_end) in enumerate(char_positions):
        if w_start <= match_start <= w_end:
            start_idx = i
        if w_start <= match_end <= w_end:
            end_idx = i
        if start_idx != -1 and end_idx != -1:
            break
    
    if start_idx == -1 or end_idx == -1:
        if DEBUG_MODE:
            print("  No valid context found")
        return ""
    
    context_start = max(0, start_idx - context_size)
    context_end = min(len(words), end_idx + context_size + 1)
    context = " ".join(words[context_start:context_end])
    
    if DEBUG_MODE:
        print(f"  Context extracted: '{context}'")
    return context

def get_context_words_all(text: str, match_start: int, match_end: int, context_size: int, case_sensitive: bool) -> str:
    """Extract context words around a match."""
    if not text:
        return ""
        
    words = text.split()
    char_count = 0
    word_positions = []
    for i, word in enumerate(words):
        word_start = char_count
        word_end = char_count + len(word)
        if not case_sensitive:
            word = word.lower()
        word_positions.append((word, word_start, word_end))
        char_count += len(word) + 1

    start_idx = end_idx = -1
    for i, (word, w_start, w_end) in enumerate(word_positions):
        if w_start <= match_start <= w_end:
            start_idx = i
        if w_start <= match_end <= w_end:
            end_idx = i
        if start_idx != -1 and end_idx != -1:
            break

    if start_idx == -1 or end_idx == -1:
        return text[max(0, match_start-20):match_end+20]

    context_start = max(0, start_idx - context_size)
    context_end = min(len(words), end_idx + context_size + 1)

    return " ".join(words[context_start:context_end])

    
def find_email_addresses(text: str, context_size: int, doc_count: int = 0, total_docs: int = 0, start_time: float = None) -> List[Tuple[int, str, str]]:
    """Find email address patterns in text with positions and contexts."""
    global TOTAL_USERS, TOTAL_HOSTS, TOTAL_COMPLETE
    
    if not text:
        if DEBUG_MODE:
            print("Text is empty, returning empty list")
        return []
    
    local_start_time = time.time()
    if DEBUG_MODE:
        print(f"Starting email search in text of length {len(text)}")
    
    if '@' not in text:
        if DEBUG_MODE:
            print("No '@' found in text, skipping regex")
        return []
    
    if DEBUG_MODE:
        print("Splitting text into words...")
    words = text.split()
    char_positions = []
    char_count = 0
    for word in words:
        w_start = char_count
        w_end = char_count + len(word)
        char_positions.append((word, w_start, w_end))
        char_count += len(word) + 1
    
    if DEBUG_MODE:
        print(f"Text split into {len(words)} words, positions computed")
    
    matches = []
    match_count = 0
    for match in EMAIL_PATTERN.finditer(text):
        match_count += 1
        email = match.group(0)
        if DEBUG_MODE and match_count % 100 == 0:
            elapsed = time.time() - local_start_time
            print(f"Processed {match_count} matches so far, elapsed time: {elapsed:.2f}s")
        
        context = get_context_words(words, char_positions, match.start(), match.end(), context_size)
        matches.append((match.start(), email, context))
        
        user_part = email.split('@')[0] if '@' in email and email.split('@')[0] else ''
        host_part = email.split('@')[1] if '@' in email and len(email.split('@')) > 1 else ''
        if user_part:
            TOTAL_USERS.add(user_part)
        if host_part:
            TOTAL_HOSTS.add(host_part)
        if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            TOTAL_COMPLETE += 1
    
    if DEBUG_MODE:
        elapsed = time.time() - local_start_time
        print(f"Found {match_count} email matches in {elapsed:.2f}s")
    else:
        if doc_count > 0 and total_docs > 0 and doc_count % 40 == 0 and start_time is not None:
            elapsed = time.time() - start_time
            print(f"({doc_count}/{total_docs}) TIME_PASSED: {elapsed:.2f}s users: {len(TOTAL_USERS)} hosts: {len(TOTAL_HOSTS)} complete: {TOTAL_COMPLETE}")
    
    matches.sort(key=lambda x: x[0])
    if DEBUG_MODE:
        print("Matches sorted by position")
    
    return matches

def find_all_matches(text: str, terms: List[str], context_size: int, case_sensitive: bool) -> List[Tuple[int, str]]:
    """Find all matches in text for given terms and return positions and contexts."""
    if not text or not terms:
        return []
    
    matches = []
    text_to_search = text if case_sensitive else text.lower()
    
    for term in terms:
        search_term = term if case_sensitive else term.lower()
        pattern = re.compile(re.escape(search_term))
        for match in pattern.finditer(text_to_search):
            context = get_context_words_all(text, match.start(), match.end(), context_size, case_sensitive)
            matches.append((match.start(), context))
    
    matches.sort(key=lambda x: x[0])
    return matches

def search_database(db_path: str, table: str, fields: List[str], return_fields: List[str], 
                   search_terms: str, case_sensitive: bool, context_size: int, show_all: bool, 
                   match_any: bool, email_search: bool = False) -> List[Dict]:
    """Search the database, store email addresses in documentMailAdresses, batch updates every 500 records."""
    global TOTAL_USERS, TOTAL_HOSTS, TOTAL_COMPLETE
    includes, excludes = parse_search_terms(search_terms) if not email_search else ([], [])
    results = []
    
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA busy_timeout = 30000")  # 30s timeout for locks
        cursor = conn.cursor()
        
        # Add documentMailAdresses column if it doesn’t exist
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [col[1] for col in cursor.fetchall()]
        if 'documentMailAdresses' not in columns:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN documentMailAdresses TEXT")
            conn.commit()
            if DEBUG_MODE:
                print("Added documentMailAdresses column to table")
        
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        total_rows = cursor.fetchone()[0]
        if DEBUG_MODE:
            print(f"Total rows in {table}: {total_rows}")
        
        all_fields = list(set(fields + [f for f in return_fields if f != '{results}']))
        select_fields = ", ".join(all_fields)
        query = f"SELECT {select_fields} FROM {table}"
        conditions = []
        params = []
        
        if not email_search and (includes or excludes):
            text_conditions = []
            for field in fields:
                field_conditions = []
                for term in includes:
                    if case_sensitive:
                        field_conditions.append(f"{field} LIKE ?")
                        params.append(f"%{term}%")
                    else:
                        field_conditions.append(f"LOWER({field}) LIKE ?")
                        params.append(f"%{term.lower()}%")
                if field_conditions:
                    text_conditions.append("(" + (" OR " if match_any else " AND ").join(field_conditions) + ")")
            if text_conditions:
                conditions.append("(" + " OR ".join(text_conditions) + ")")
            
            for field in fields:
                for term in excludes:
                    if case_sensitive:
                        conditions.append(f"{field} NOT LIKE ?")
                        params.append(f"%{term}%")
                    else:
                        conditions.append(f"LOWER({field}) NOT LIKE ?")
                        params.append(f"%{term.lower()}%")
        else:
            if VERBOSE_MODE:
                print(f"Setting for email_search is {email_search}. " + "There are no includes or excludes." if not includes or excludes else "Therefore there are no extra conditions set for selection." + "So no conditions are set." if not conditions else f"Even though {len(conditions)} are set."  )
                
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        if email_search and not conditions:
                query +=  " WHERE documentMailAdresses IS NULL"
                
        if DEBUG_MODE:
            if len(params) > 0:
                print(f"Executing query: {query} with params: {params}")
            else:
                print(f"Executing query: {query}")
        
        if len(params) > 0:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        all_rows = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        row_count = 0
        
        if email_search:
            cursor.execute(f"SELECT COUNT(*) FROM documents_archives WHERE documentMailAdresses IS NOT NULL")
            with_mailaddress = cursor.fetchone()[0]
            print(f"There are {total_rows} records in the database of which {with_mailaddress} might mention email addresses.\n")
            cursor.execute(query)
            
        start_time = time.time()
        batch_updates = []
        
        for row in all_rows:
            row_count += 1
            row_dict = dict(zip(column_names, row))
            search_text = " ".join(str(row_dict.get(field, '')) for field in fields if row_dict.get(field))
            
            if search_text:
                if email_search:
                    matches = find_email_addresses(search_text, context_size, row_count, total_rows, start_time)

                    if matches:
                        if VERBOSE_MODE:
                            print(f"Found {len(matches)} matches in row {row_count}...")
                        doc_name = row_dict.get('documentName', '')
                        doc_parts = split_document_name(doc_name)
                        in_documents, production_date, locatie, title = lookup_in_documents(
                            conn,
                            doc_parts['indicatorRequestNumber'],
                            doc_parts['indicatorDocumentID']
                        )
                        result_entry = {
                            'matches': matches,
                            'fields': {field: row_dict.get(field, '') for field in return_fields if field != '{results}'},
                            'inDocuments': in_documents,
                            'productionDate': production_date,
                            'locatie': locatie,
                            'title': title
                        }
                        if DEBUG_MODE:
                            print(result_entry)
                            
                        results.append(result_entry)
                        
                        email_json = json.dumps([{
                            'position': pos,
                            'email': email,
                            'context': context
                        } for pos, email, context in matches])
                        batch_updates.append((email_json, doc_name))
                
                else:
                    matches = find_all_matches(search_text, includes, context_size, case_sensitive) if includes else []
                    
                
            # Batch update SQLite every 500 records
            if row_count % 500 == 0 and batch_updates:
                try:
                    print(f"Updating database after processing 500 records with {len(batch_updates)} documents having results")
                        
                    cursor.executemany(f"UPDATE {table} SET documentMailAdresses = ? WHERE documentName = ?", batch_updates)
                    conn.commit()
                    if VERBOSE_MODE:
                        print(f"Updated {len(batch_updates)} records in documentMailAdresses at row {row_count}")
                except sqlite3.Error as e:
                    print(f"SQLite error during batch update at row {row_count}: {e}")
                    break
                batch_updates = []
        
        # Final batch update
        if batch_updates:
            try:
                print(f"Updating database with final batch of {len(batch_updates)} documents with results")
                cursor.executemany(f"UPDATE {table} SET documentMailAdresses = ? WHERE documentName = ?", batch_updates)
                conn.commit()
                report_email_stats
                if DEBUG_MODE:
                    print(f"Final update: {len(batch_updates)} records in documentMailAdresses")
            except sqlite3.Error as e:
                print(f"SQLite error during final batch update: {e}")
        
    if DEBUG_MODE:
        print(f"Processed {row_count} rows, found {len(results)} matching documents")
    
    return results
    
def sanitize_string(value: str) -> str:
    """Remove or replace illegal characters for Excel compatibility."""
    if not isinstance(value, str):
        return str(value)
    illegal_chars = re.compile(r'[\x00-\x08\x0B-\x0C\x0E-\x1F&<>$]')
    return illegal_chars.sub(' ', value).strip()

def save_to_excel(results: List[Dict], return_fields: List[str], show_all: bool, search_terms: str) -> str:
    """Save results to Excel file and return filename."""
    total_results = sum(len(r['matches']) for r in results)
    unique_files = len(set(r['fields'].get('documentName', '') for r in results if 'documentName' in r['fields']))
    date_str = datetime.now().strftime("%Y%m%d")
    search_words = search_terms.replace(',', '_').replace(' ', '_') if search_terms else "emailsearch"
    filename = f"{date_str}_{total_results}_{unique_files}_{search_words}.xlsx"
    
    base_fields = [field for field in return_fields if field != '{results}']
    headers = ['Result', 'MatchesInDoc'] + base_fields
    if show_all:
        headers += ['inDocuments', 'position', 'results']
    else:
        headers += ['results']
    headers += ['productionDate', 'documentURL', 'title']
    
    data = []
    result_counter = 0
    for result in results:
        matches = result['matches']
        num_matches = len(matches)
        
        if show_all:
            for match_data in matches:
                if len(match_data) == 3:  # Email search: (position, email, context)
                    position, match_text, _ = match_data
                else:  # Regular search: (position, match_text)
                    position, match_text = match_data
                result_counter += 1
                row = [f"({result_counter}/{total_results})", num_matches]
                for field in base_fields:
                    row.append(sanitize_string(result['fields'].get(field, '')))
                row.extend([sanitize_string(result['inDocuments']), position, sanitize_string(match_text)])
                row.extend([sanitize_string(result['productionDate']), sanitize_string(result['locatie']), sanitize_string(result['title'])])
                data.append(row)
        else:
            result_counter += 1
            row = [f"({result_counter}/{total_results})", num_matches]
            for field in base_fields:
                row.append(sanitize_string(result['fields'].get(field, '')))
            match_text = matches[0][1]  # First match or email
            row.append(sanitize_string(match_text))
            row.extend([sanitize_string(result['productionDate']), sanitize_string(result['locatie']), sanitize_string(result['title'])])
            data.append(row)
    
    df = pd.DataFrame(data, columns=headers)
    writer = pd.ExcelWriter(filename, engine='openpyxl')
    df.to_excel(writer, index=False, sheet_name='Results')
    
    worksheet = writer.sheets['Results']
    url_col_idx = headers.index('documentURL') + 1
    for row in range(2, len(df) + 2):
        cell = worksheet.cell(row=row, column=url_col_idx)
        if cell.value and cell.value != "N/A" and cell.value.startswith('http'):
            cell.hyperlink = cell.value
            cell.style = 'Hyperlink'
    
    writer.close()
    return filename

def create_zip_from_results(results: List[Dict], search_terms: str, no_duplicates=False) -> str:
    """Create a ZIP file from documents in results without deleting tmp folder."""
    total_results = sum(len(r['matches']) for r in results)
    unique_docs = set((r['fields'].get('fromFilePath', ''), r['fields'].get('documentName', '')) for r in results)
    date_str = datetime.now().strftime("%Y%m%d")
    search_words = search_terms.replace(',', '_').replace(' ', '_') if search_terms else "emailsearch"
    datetime_str = datetime.now().strftime("%Y%m%d%H%M%S_") + search_words
    zip_filename = f"{date_str}_{total_results}_{len(unique_docs)}_{search_words}.zip"

    zip_files = set(r['fields'].get('fromFilePath', '') for r in results if r['fields'].get('fromFilePath', '').endswith('.zip'))
    if not zip_files:
        print("No valid ZIP files found in results.")
        return ""

    tmp_dir = f'./tmp/{datetime_str}'
    try:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)
    except (OSError, PermissionError) as e:
        print(f"Error creating temporary directory {tmp_dir}: {e}")
        return ""

    extracted_files = {}
    for zip_path in zip_files:
        if not os.path.exists(zip_path):
            print(f"Warning: ZIP file not found: {zip_path}")
            continue
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                docs_to_extract = [r['fields'].get('documentName', '') for r in results 
                                  if r['fields'].get('fromFilePath', '') == zip_path]
                for doc_name in docs_to_extract:
                    if doc_name in zip_ref.namelist():
                        base_name, ext = os.path.splitext(doc_name)
                        target_name = doc_name
                        counter = 1
                        while target_name in extracted_files:
                            if no_duplicates:
                                print(f"Skipping duplicate {doc_name} from {zip_path} due to --noduplicates")
                                break
                            target_name = f"{base_name}_{counter}{ext}"
                            counter += 1
                        else:
                            extracted_files[target_name] = True
                            with zip_ref.open(doc_name) as source, open(os.path.join(tmp_dir, target_name), 'wb') as target:
                                shutil.copyfileobj(source, target)
        except zipfile.BadZipFile:
            print(f"Error: {zip_path} is not a valid ZIP file.")
            continue
        except Exception as e:
            print(f"Error extracting from {zip_path}: {e}")
            continue

    extracted_count = 0
    try:
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as new_zip:
            for root, _, files in os.walk(tmp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, tmp_dir)
                    new_zip.write(file_path, arcname)
                    extracted_count += 1
        if extracted_count == 0:
            print("Warning: No files were added to the ZIP.")
        else:
            print(f"Added {extracted_count} files to {zip_filename}")
    except Exception as e:
        print(f"Error creating ZIP {zip_filename}: {e}")
        return ""

    print(f"Temporary files left in {tmp_dir} for other use")
    return zip_filename

def report_email_stats(results: List[Dict]):
    """Report statistics on email addresses found in results."""
    unique_docs = set(r['fields'].get('documentName', '') for r in results if 'documentName' in r['fields'])
    all_emails = []
    for result in results:
        for _, email, _ in result['matches']:
            all_emails.append(email)
    
    complete_emails = [e for e in all_emails if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', e)]
    incomplete_hostnames = [e for e in all_emails if e.startswith('@') and not re.match(r'^[a-zA-Z0-9._%+-]+@', e)]
    incomplete_usernames = [e for e in all_emails if e.endswith('@') and not re.match(r'@.+\..+$', e)]
    
    print(f"\nEmail Address Statistics:")
    print(f"Unique records with email addresses: {len(unique_docs)}")
    print(f"Complete email addresses found: {len(complete_emails)}")
    print(f"Incomplete usernames (e.g., username@): {len(incomplete_usernames)}")
    print(f"Incomplete hostnames (e.g., @hostname.tld): {len(incomplete_hostnames)}")
    
    print("\nTop 10 Complete Email Addresses:")
    for email, count in Counter(complete_emails).most_common(5):
        print(f"  {email}: {count} occurrences")
    
    print("\nTop 10 Incomplete Usernames:")
    for email, count in Counter(incomplete_usernames).most_common(5):
        print(f"  {email}: {count} occurrences")
    
    print("\nTop 10 Incomplete Hostnames:")
    for email, count in Counter(incomplete_hostnames).most_common(5):
        print(f"  {email}: {count} occurrences")
        

def format_output(results: List[Dict], return_fields: List[str], show_all: bool, save: bool, create: bool, search_terms: str, email_search: bool) -> None:
    """Format and print results, saving to Excel if --save is specified."""
    global TOTAL_USERS, TOTAL_HOSTS, TOTAL_COMPLETE
    print(" ".join(sys.argv))
    
    total_results = sum(len(r['matches']) for r in results)
    unique_files = len(set(r['fields'].get('documentName', '') for r in results if 'documentName' in r['fields']))
    print(f"Total results: {total_results}, Unique files: {unique_files}")
    
    if not show_all:
        print("Showing only unique documents (use --all to see all matches)")
    
    if email_search:
        report_email_stats(results)
    
    if save:
        email_data = []
        for result in results:
            doc_name = result['fields'].get('documentName', '')
            for pos, email, context in result['matches']:
                email_data.append({
                    'Document': doc_name,
                    'Position': pos,
                    'Email': email,
                    'Context': context
                })
        if email_data:
            df = pd.DataFrame(email_data)
            df.to_excel("email_matches.xlsx", index=False)
            print(f"Saved {len(email_data)} email matches to email_matches.xlsx")
    
    if create:
        zip_filename = create_zip_from_results(results, search_terms, no_duplicates=args.noduplicates)
        print(f"Documents archived to {zip_filename}")
    
    if save or create or email_search:
        return
    
    result_counter = 0
    for doc_idx, result in enumerate(results, 1):
        matches = result['matches']
        num_matches = len(matches)
        
        if show_all:
            for match_idx, match_data in enumerate(matches, 1):
                if len(match_data) == 3:
                    position, match_text, _ = match_data
                else:
                    position, match_text = match_data
                result_counter += 1
                field_values = []
                for field in return_fields:
                    if field == '{results}':
                        field_values.append(f'"{result["inDocuments"]}"')
                        field_values.append(f"{position}")
                        field_values.append(f'"{match_text}"')
                    else:
                        field_values.append(f'"{result["fields"].get(field, "")}"')
                field_values.append(f'"{result["productionDate"]}"')
                field_values.append(f'"{result["locatie"]}"')
                field_values.append(f'"{result["title"]}"')
                output = f"({result_counter}/{total_results});{num_matches};" + ";".join(field_values)
                print(output)
        else:
            result_counter += 1
            field_values = []
            match_text = matches[0][1]
            for field in return_fields:
                if field == '{results}':
                    field_values.append(f'"{match_text}"')
                else:
                    field_values.append(f'"{result["fields"].get(field, "")}"')
            field_values.append(f'"{result["productionDate"]}"')
            field_values.append(f'"{result["locatie"]}"')
            field_values.append(f'"{result["title"]}"')
            output = f"({result_counter}/{total_results});{num_matches};" + ";".join(field_values)
            print(output)

def main():
    global args, DEBUG_MODE, VERBOSE_MODE, TOTAL_USERS, TOTAL_HOSTS, TOTAL_COMPLETE
    parser = argparse.ArgumentParser(description='Search documents archives')
    parser.add_argument('--db', required=True, help='Database path')
    parser.add_argument('--table', required=True, help='Table name')
    parser.add_argument('--fields', required=True, help='Comma-separated list of fields to search')
    parser.add_argument('--return_fields', required=True, help='Comma-separated list of fields to return')
    parser.add_argument('--search', help='Search terms (comma-separated, use -term to exclude)')
    parser.add_argument('--case', action='store_true', help='Case sensitive search')
    parser.add_argument('--context', type=int, default=3, help='Number of context words')
    parser.add_argument('--all', action='store_true', help='Show all matches instead of just first')
    parser.add_argument('--save', action='store_true', help='Save email matches to Excel file')
    parser.add_argument('--create', action='store_true', help='Create ZIP file with matching documents')
    parser.add_argument('--any', action='store_true', help='Match any search term across all fields (OR search)')
    parser.add_argument('--noduplicates', action='store_true', help='Skip renaming and processing of duplicate filenames in ZIP creation')
    parser.add_argument('--emailaddress', action='store_true', help='Search for email addresses instead of using search terms')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    DEBUG_MODE = args.debug
    VERBOSE_MODE = args.verbose
    
    if args.emailaddress and args.search:
        print("Error: --emailaddress cannot be used with --search. Use one or the other.")
        sys.exit(1)
    
    search_fields = [f.strip() for f in args.fields.split(',')]
    return_fields = [f.strip() for f in args.return_fields.split(',')]
    
    TOTAL_USERS.clear()
    TOTAL_HOSTS.clear()
    TOTAL_COMPLETE = 0
    
    results = search_database(
        args.db, args.table, search_fields, return_fields,
        args.search, args.case, args.context, args.all, args.any, args.emailaddress
    )
    
    format_output(results, return_fields, args.all, args.save, args.create, args.search, args.emailaddress)

if __name__ == "__main__":
    main()

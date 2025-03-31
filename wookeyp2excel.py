import pandas as pd
import argparse
import sys
import os
import re
import sqlite3

VERBOSE_MODE = False

def parse_text_to_excel(input_file, output_file, db_file=None, length_check=None, test=False):
    columns = ['Nr.', 'fromfile', 'Omschrijving', 'Beoordeling', 'Weigeringsgrond(en)']
    valid_beoordeling = {'Openbaar', 'Reeds Openbaar', 'Deels Openbaar', 'Niet Openbaar'}

    def clean_text(text):
        cleaned = text.replace('\xa0', ' ').replace('\u2010', '-').replace('\u2013', '-').replace('\u2014', '-')
        cleaned = re.sub(r'[\u200b-\u200f\u2028-\u202f]', ' ', cleaned)
        cleaned = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', cleaned)
        return re.sub(r'\s+', ' ', cleaned).strip()

    def check_beoordeling_at_end(omschrijving_text):
        if not omschrijving_text:
            return None, omschrijving_text
        omschrijving_cleaned = clean_text(omschrijving_text)
        parts = omschrijving_cleaned.split()
        if len(parts) >= 2 and " ".join(parts[-2:]) in valid_beoordeling:
            return " ".join(parts[-2:]), " ".join(parts[:-2]).strip()
        if parts[-1] in valid_beoordeling:
            return parts[-1], " ".join(parts[:-1]).strip()
        return None, omschrijving_text

    # Read the file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # Determine if the file has any Beoordeling
    NO_Beoordeling = True
    for line in lines:
        if clean_text(line) in valid_beoordeling:
            NO_Beoordeling = False
            break
    if VERBOSE_MODE:
        print(f"NO_Beoordeling set to: {NO_Beoordeling}")

    # Find data start
    data_start_idx = 0
    for i, line in enumerate(lines):
        first_part = line.split(' ', 1)[0] if ' ' in line else line
        if first_part.isdigit() and not first_part.startswith('0'):
            data_start_idx = i
            break
        if i > 10:
            raise ValueError("Could not find start of data")

    data = []
    i = data_start_idx
    nr = None
    omschrijving = ""
    beoordeling = None
    weigeringsgronden = ""
    previous_nr = 0
    line_count = data_start_idx
    fromfile = input_file

    # First pass: Current logic
    while i < len(lines) and (not test or len(data) < 100):
        line = lines[i]
        if VERBOSE_MODE:
            print(f"Processing line {i}: '{line}'")
        if line.startswith('--- Page'):
            i += 1
            continue
        first_part = line.split(' ', 1)[0] if ' ' in line else line
        is_standalone_number = first_part.isdigit() and not first_part.startswith('0') and ' ' not in line
        line_starts_with_number = first_part.isdigit() and not first_part.startswith('0')

        if NO_Beoordeling:
            if is_standalone_number:
                current_nr = int(first_part)
                if length_check and len(first_part) != length_check:
                    if VERBOSE_MODE:
                        print(f"Skipping {current_nr}: length mismatch")
                    i += 1
                    continue
                if nr and omschrijving:
                    data.append([nr, fromfile, omschrijving, None, None])
                    if VERBOSE_MODE:
                        print(f"Added record: {nr}, {previous_nr}, {current_nr}, None, {omschrijving}, None")
                nr = first_part
                previous_nr = current_nr if 'current_nr' in locals() else 0
                omschrijving = ""
                beoordeling = None
                weigeringsgronden = ""
                i += 1
            elif nr:
                cleaned_line = clean_text(line)
                if not omschrijving:
                    omschrijving = cleaned_line
                elif omschrijving.endswith('-'):
                    omschrijving += cleaned_line
                else:
                    omschrijving += " " + cleaned_line
                i += 1
            else:
                i += 1
        else:
            if is_standalone_number:
                current_nr = first_part
                if length_check and len(first_part) != length_check:
                    if VERBOSE_MODE:
                        print(f"Skipping {current_nr}: length mismatch")
                    i += 1
                    continue
                if nr and beoordeling:  # Save previous record if complete
                    data.append([nr, fromfile, omschrijving, beoordeling, weigeringsgronden])
                    if VERBOSE_MODE:
                        print(f"Added record: {nr}, {previous_nr}, {current_nr}, {beoordeling}, {omschrijving}, {weigeringsgronden}")
                nr = first_part
                previous_nr = current_nr if 'current_nr' in locals() else 0
                omschrijving = ""
                beoordeling = None
                weigeringsgronden = ""
                i += 1
            elif nr:
                cleaned_line = clean_text(line)
                if cleaned_line in valid_beoordeling and not beoordeling:
                    beoordeling = cleaned_line
                    i += 1
                elif beoordeling and not weigeringsgronden:
                    weigeringsgronden = cleaned_line
                    i += 1
                else:
                    i += 1  # Skip unexpected lines
            else:
                i += 1

    if nr and (omschrijving or (not NO_Beoordeling and (beoordeling or weigeringsgronden))):
        data.append([nr, fromfile, omschrijving, None if NO_Beoordeling else beoordeling, None if NO_Beoordeling else weigeringsgronden])
        if VERBOSE_MODE:
            print(f"({line_count}) Added final record: {nr}, {None if NO_Beoordeling else beoordeling}, {omschrijving}, {None if NO_Beoordeling else weigeringsgronden}")

    # Second pass: New format if no data found
    if not data:
        if VERBOSE_MODE:
            print("No data found with first pass, attempting second pass for new format")
        i = data_start_idx
        nr = None
        omschrijving = ""
        beoordeling = None
        weigeringsgronden = ""
        previous_nr = 0

        while i < len(lines) and (not test or len(data) < 100):
            line = lines[i]
            if VERBOSE_MODE:
                print(f"Second pass - Processing line {i}: '{line}'")
            if line.startswith('--- Page'):
                i += 1
                continue
            first_part = line.split(' ', 1)[0] if ' ' in line else line
            line_starts_with_number = first_part.isdigit() and not first_part.startswith('0')

            if line_starts_with_number:
                current_nr = first_part
                if length_check and len(first_part) != length_check:
                    if VERBOSE_MODE:
                        print(f"Skipping {current_nr}: length mismatch")
                    i += 1
                    continue
                if nr and beoordeling:  # Save previous record if complete
                    data.append([nr, fromfile, omschrijving, beoordeling, weigeringsgronden])
                    if VERBOSE_MODE:
                        print(f"Added record: {nr}, {previous_nr}, {current_nr}, {beoordeling}, {omschrijving}, {weigeringsgronden}")
                nr = first_part
                previous_nr = current_nr if 'current_nr' in locals() else 0
                omschrijving = clean_text(line.split(' ', 1)[1]) if ' ' in line else ""
                beoordeling = None
                weigeringsgronden = ""
                i += 1
            elif nr:
                cleaned_line = clean_text(line)
                if cleaned_line in valid_beoordeling and not beoordeling:
                    beoordeling = cleaned_line
                    i += 1
                elif beoordeling and not weigeringsgronden:
                    weigeringsgronden = cleaned_line
                    i += 1
                else:
                    i += 1  # Skip unexpected lines
            else:
                i += 1

        if nr and (omschrijving or (beoordeling or weigeringsgronden)):
            data.append([nr, fromfile, omschrijving, beoordeling, weigeringsgronden])
            if VERBOSE_MODE:
                print(f"({line_count}) Added final record: {nr}, {beoordeling}, {omschrijving}, {weigeringsgronden}")

    if not data:
        raise ValueError("No valid data entries found in the file after both passes.")

    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    df['Nr.'] = df['Nr.'].astype(str)

    # Save to Excel
    df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"Excel file successfully created: '{output_file}' with {len(df)} entries.")
    
    # Save to Database if db_file is provided
    if db_file:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # Create table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents_wobcovid19 (
                Id INTEGER PRIMARY KEY AUTOINCREMENT,
                nr TEXT,
                fromfile TEXT,
                omschrijving TEXT,
                beoordeling TEXT,
                weigeringsgronden TEXT,
                UNIQUE(nr, fromfile)
            )
        ''')

        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_nr ON documents_wobcovid19 (nr)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_fromfile ON documents_wobcovid19 (fromfile)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_omschrijving ON documents_wobcovid19 (omschrijving)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_beoordeling ON documents_wobcovid19 (beoordeling)')

        # Rename DataFrame columns to match database schema
        df_db = df.rename(columns={'Nr.': 'nr', 'Omschrijving': 'omschrijving', 'Beoordeling': 'beoordeling', 'Weigeringsgrond(en)': 'weigeringsgronden'})

        # Check existing records based on nr and fromfile
        existing_df = pd.read_sql_query("SELECT nr, fromfile FROM documents_wobcovid19", conn)
        if not existing_df.empty:
            print(f"Found {len(existing_df)} existing records")
            # Identify records to update (existing nr and fromfile)
            df_to_update = df_db.merge(existing_df, on=['nr', 'fromfile'], how='inner')
            if not df_to_update.empty:
                for index, row in df_to_update.iterrows():
                    cursor.execute('''
                        UPDATE documents_wobcovid19
                        SET omschrijving = ?, beoordeling = ?, weigeringsgronden = ?
                        WHERE nr = ? AND fromfile = ?
                    ''', (row['omschrijving'], row['beoordeling'], row['weigeringsgronden'], row['nr'], row['fromfile']))
                print(f"Updated {len(df_to_update)} existing entries in database: '{db_file}'")

            # Identify new records (not in existing)
            df_to_insert = df_db.merge(existing_df, on=['nr', 'fromfile'], how='left', indicator=True)
            df_to_insert = df_to_insert[df_to_insert['_merge'] == 'left_only'].drop(columns=['_merge'])
        else:
            df_to_insert = df_db  # All records are new if no existing records

        # Insert new data into the table
        if not df_to_insert.empty:
            df_to_insert.to_sql('documents_wobcovid19', conn, if_exists='append', index=False)
            print(f"Added {len(df_to_insert)} new entries to database: '{db_file}' in table 'documents_wobcovid19'.")
        else:
            print(f"No new entries to add to database: '{db_file}' in table 'documents_wobcovid19'.")

        conn.commit()
        conn.close()
        
def main():
    global VERBOSE_MODE
    parser = argparse.ArgumentParser(description="wookeyp2excel.py - Convert a multi-line text file with document data into an Excel file and optionally a database.")
    parser.add_argument("input_file", help="Path to the input text file")
    parser.add_argument("--output", help="Path to the output Excel file (default: input_file.xlsx)", default=None)
    parser.add_argument("--db", help="Path to the SQLite database file (optional)", default=None)
    parser.add_argument("--verbose", action="store_true", help="Verbose mode ON")
    parser.add_argument("--test", action="store_true", help="Test 50 lines")
    parser.add_argument("--length", type=int, help="Expected length of Nr. values (e.g., 6 or 7).")
    args = parser.parse_args()
    VERBOSE_MODE = args.verbose
    if args.output is None:
        args.output = f"{os.path.splitext(args.input_file)[0]}.xlsx"
    try:
        parse_text_to_excel(args.input_file, args.output, db_file=args.db, length_check=args.length, test=args.test)
    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

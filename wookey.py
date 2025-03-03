import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs, urlunparse, urlencode, unquote
from concurrent.futures import ThreadPoolExecutor
import sys
import time
from datetime import datetime
import os
import shutil
import math
import argparse
import subprocess
import re
import sqlite3
import json
from typing import Dict
from collections import Counter
from tenacity import retry, stop_after_attempt, wait_exponential
import inspect
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.firefox import GeckoDriverManager
import functools
from functools import lru_cache
import platform
from io import StringIO
import uuid
from uuid import uuid4
import signal
import threading
import concurrent.futures
from multiprocessing import Manager
from time import sleep
import random
from threading import Lock
from selenium.common.exceptions import TimeoutException

## Script Version - This is a major release
SCRIPT_VERSION = "7.0.0"

## Defaults
VERBOSE_MODE = False
SILENT_MODE = False
DEBUG_MODE = False
FORCE_MODE = False
WRITE_RESULTS = True
WRITE_DEBUG_RESULTS = False

## Global configuration
# 20250119, replaced base_url = "https://open.minvws.nl/zoeken?sort=_score&sortorder=desc&doctype%5B%5D=dossier&doctype%5B%5D=dossier.publication"
# base_url = "https://open.minvws.nl/zoeken?sort=_score&sortorder=desc&doctype%5B0%5D=dossier&doctype%5B1%5D=dossier.publication"
base_url = "https://open.minvws.nl/zoeken?doctype%5B0%5D=dossier&doctype%5B1%5D=dossier.publication&sort=publication_date&sortorder=desc"
# base_url = "https://open.minvws.nl/zoeken?doctype%5B0%5D=dossier&doctype%5B1%5D=dossier.publication&sort=_score&sortorder=desc"

base_href = "https://open.minvws.nl/"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
# Global variable for command-line arguments
args = None
# Selenium driver
driver = None

# Local folders
DOWNLOAD_INVENTARIS = "inventaris_files"
DOWNLOAD_BESLUIT = "besluit_files"
DOWNLOAD_DOCUMENTEN = "documenten_files"

# Inventaris Excel files, headers
INVENTARISFILE_HEADERS = "Document ID,	Document naam,	Bestandsnaam,	Beoordeling,	Beoordelingsgrond,	Toelichting,	Publieke link,	Locatie open.minvws.nl,	Opgeschort,	Definitief ID"

# cached tag data
tag_cache = Manager().dict() if __name__ == "__main__" else {}
tag_cache_lock = threading.Lock()

stop_threads = False

def signal_handler(signum, frame):
    global stop_threads
    stop_threads = True

    safe_print("Signal handler caught interrupt. Exiting gracefully...")
    # Ensure cleanup here or call cleanup()
    try:
        executor.shutdown(wait=False)
    except NameError:
        pass  # If executor isn't defined in this scope
    
    try:
        conn.close()
    except Exception:
        pass

    try:
        driver.quit()
    except NameError:
        pass
    
    cleanup()
    sys.exit(0)  

signal.signal(signal.SIGINT, signal_handler)

print_lock = threading.Lock()

def safe_print(*args, force=False, **kwargs):
    with print_lock:
        if SILENT_MODE and not force:
            return
        else:
            if SILENT_MODE and force:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]", *args, **kwargs, flush=True)
            else:
                print(*args, **kwargs, flush=True)
        
# START TABLE CONFIGURATION
# Load table configurations from JSON file
with open('data.config', 'r') as config_file:
    TABLE_CONFIGS = json.load(config_file)

def print_message(message, is_debug=False):
    """Print messages based on the verbose and quiet settings."""
    if not args.quiet and (VERBOSE_MODE or not is_debug):
        print(message)

def printm(message, verbose):
    """Prints message if verbose mode is enabled."""
    if verbose:
        print(message)
        
def cleanup():
    global driver
    if driver:
        driver.quit()
        driver = None

def create_folders():
    folders = [DOWNLOAD_INVENTARIS, DOWNLOAD_BESLUIT, DOWNLOAD_DOCUMENTEN]
    
    for folder in folders:
        try:
            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f"Created folder: {folder}")
        except OSError as error:
            print(f"Error creating directory {folder}: {error}")

def create_connection(db_file, db_platform='sqlite3'):
    if VERBOSE_MODE:
        callerframerecord = inspect.stack()[1]
        frame = callerframerecord[0]
        info = inspect.getframeinfo(frame)
        lineno = info.lineno
        print(f"DEBUG: {create_connection.__name__} at line {lineno}")
        
    """Create a database connection to a SQLite database."""
    try:
        if db_platform.lower() == 'sqlite3':
            conn = sqlite3.connect(db_file, isolation_level=None, check_same_thread=False)
        else:
            raise ValueError(f"Unsupported database platform: {db_platform}")
        if VERBOSE_MODE:
            print(f"Connected to {db_platform} database: {db_file}")
        return conn
    except Exception as e:
        if VERBOSE_MODE:
            print(f"Database error: {e}")
        sys.exit(1)

def retry_execute(cursor, sql, params=(), retries=3, delay=5, verbose=False):
    for attempt in range(retries):
        try:
            if verbose:
                print(sql, params)
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            return cursor
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                if verbose:
                    print(f"Database is locked, retrying in {delay} seconds. Attempt {attempt+1} of {retries}.")
                time.sleep(delay)
            else:
                raise
    raise sqlite3.OperationalError("Failed to execute SQL after retries: " + sql)

def create_table_if_not_exists(conn, config):
    """Create all tables and their indexes if they do not exist based on the configuration."""
    cursor = conn.cursor()
    
    for table_config in config:
        table_name = table_config['name']
        columns = table_config['columns']
        primary_key = table_config.get('primary_key', None)
        
        # Prepare column definitions
        column_defs = ', '.join(f'"{col}" {col_def["type"]}' for col, col_def in columns.items())
        primary_key_clause = f", PRIMARY KEY ({', '.join(f'{col}' for col in primary_key)})" if primary_key else ""
        
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS '{table_name}' (
            {column_defs}
            {primary_key_clause}
        )
        """
        
        try:
            cursor.execute(create_table_sql)
            if VERBOSE_MODE:
                print(f"Checked and created if necessary: {table_name}")

            # Create indexes
            for index_key, index_columns in table_config.items():
                if index_key.startswith('index_'):
                    index_name = f"{table_name}_{'_'.join([col.replace(' ', '_') for col in index_columns if col])}"
                    if index_columns and index_columns[0]:  # Ensure we have columns to index
                        index_sql = f"CREATE INDEX IF NOT EXISTS {index_name} ON \"{table_name}\"({', '.join(f'{col}' for col in index_columns)});"
                        cursor.execute(index_sql)
                        if VERBOSE_MODE:
                            print(f"Created index {index_name} for table {table_name}")
                    else:
                        if VERBOSE_MODE:
                            print(f"Skipping empty index creation for {table_name}")

        except sqlite3.OperationalError as e:
            if VERBOSE_MODE:
                print(f"Error creating or updating table {table_name}: {e}")
            conn.rollback()
        except Exception as e:
            print(f"Unexpected error with table {table_name}: {e}")
            conn.rollback()

    conn.commit()

def quote_table_name(name):
    return f'"{name.replace('"', '""')}"'

def quote_field_name(name):
    return f'"{name.replace('"', '""')}"'
    
def ensure_tables_exist(conn, table_configs):
    cursor = conn.cursor()
    
    for table_config in table_configs:
        table_name = table_config['name']
        columns = table_config['columns']
        primary_key = table_config.get('primary_key', None)

        if VERBOSE_MODE:
            print(f"Debug: Checking table: {table_name}")

        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
        table_exists = cursor.fetchone()

        if table_exists:
            if VERBOSE_MODE:
                print(f"Debug: Table '{table_name}' already exists. Checking for schema changes.")

            cursor.execute(f"PRAGMA table_info({table_name})")
            existing_schema = cursor.fetchall()

            cursor.execute(f"PRAGMA index_list({quote_table_name(table_name)})")
            index_list = cursor.fetchall()
            indexed_columns = set()
            for index in index_list:
                index_name = quote_field_name(index[1])
                cursor.execute(f"PRAGMA index_info({index_name})")
                for column in cursor.fetchall():
                    indexed_columns.add(column[2])

            db_fields = [col[1] for col in existing_schema]
            config_fields = list(columns.keys())
            rename_mappings = {}
            fields_to_remove = []

            if VERBOSE_MODE:
                print(f"Debug: Schema Comparison for {table_name}:")
                print(f"Database Fields: {', '.join(db_fields)}")
                print(f"Configuration Fields: {', '.join(config_fields)}")

            for db_index, db_field in enumerate(db_fields):
                if db_index < len(config_fields):
                    config_field = config_fields[db_index]
                    db_col_type = next((col[2] for col in existing_schema if col[1] == db_field), None)
                    config_col_type = columns[config_field]['type']
                    if db_col_type == config_col_type:
                        rename_mappings[db_field] = config_field
                        if VERBOSE_MODE:
                            print(f"Field {db_index + 1}: '{db_field}{'*' if db_field in indexed_columns else ''}' (Database) -> '{config_field}' (Configuration) - Renamed")
                    else:
                        if VERBOSE_MODE:
                            print(f"Field {db_index + 1}: '{db_field}{'*' if db_field in indexed_columns else ''}' (Database) does not match '{config_field}' (Configuration)")
                else:
                    fields_to_remove.append(db_field)
                    if VERBOSE_MODE:
                        print(f"Extra field in database to be removed: {db_field}{'*' if db_field in indexed_columns else ''}")

            if fields_to_remove:
                if VERBOSE_MODE:
                    print(f"Debug: Attempting to remove extra fields: {', '.join(fields_to_remove)}")
                temp_table_name = f"{table_name}_temp"
                try:
                    with conn:
                        temp_cols = ', '.join(f'"{col[1]}" {col[2]}' for col in existing_schema if col[1] not in fields_to_remove)
                        cursor.execute(f"CREATE TABLE {temp_table_name} ({temp_cols})")
                        cols_to_copy = ', '.join(f'"{col[1]}"' for col in existing_schema if col[1] not in fields_to_remove)
                        cursor.execute(f"INSERT INTO {temp_table_name} SELECT {cols_to_copy} FROM {table_name}")
                        cursor.execute(f"DROP TABLE {table_name}")
                        cursor.execute(f"ALTER TABLE {temp_table_name} RENAME TO {table_name}")
                        if VERBOSE_MODE:
                            print(f"Debug: Successfully removed extra fields from {table_name}")
                except sqlite3.OperationalError as e:
                    print(f"Error removing extra fields from {table_name}: {e}")

            if rename_mappings:
                if VERBOSE_MODE:
                    print("Debug: Renaming fields:")
                for old_name, new_name in rename_mappings.items():
                    if old_name != new_name:
                        try:
                            cursor.execute(f"ALTER TABLE {table_name} RENAME COLUMN `{old_name}` TO `{new_name}`;")
                            if VERBOSE_MODE:
                                print(f"Debug: Successfully renamed '{old_name}' to '{new_name}'")
                        except sqlite3.OperationalError as e:
                            print(f"Error renaming column '{old_name}' to '{new_name}': {e}")

            for config_index, config_field in enumerate(config_fields):
                if config_index >= len(db_fields) or config_field not in [new_name for old_name, new_name in rename_mappings.items()]:
                    if VERBOSE_MODE:
                        print(f"Debug: Checking existence of field {config_field}:")
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    new_schema = cursor.fetchall()
                    if not any(col[1] == config_field for col in new_schema):
                        if VERBOSE_MODE:
                            print(f"Debug: Adding new column '{config_field}'")
                        try:
                            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN `{config_field}` {columns[config_field]['type']};")
                            if VERBOSE_MODE:
                                print(f"Debug: Added column '{config_field}' to table '{table_name}'")
                        except sqlite3.OperationalError as e:
                            print(f"Error adding column '{config_field}' to '{table_name}': {e}")

        else:
            # Create new table if it does not exist
            column_defs = ', '.join(f'"{col_name}" {col_def["type"]}' for col_name, col_def in columns.items())
            primary_key_clause = ""
            if primary_key:
                primary_key_clause = f", PRIMARY KEY ({', '.join(f'`{pk}`' for pk in primary_key)})"

            create_table_sql = f"""
            CREATE TABLE '{table_name}' (
                {column_defs}
                {primary_key_clause}
            )
            """
            if VERBOSE_MODE:
                print(f"Debug: SQL to create table: {create_table_sql}")
            try:
                cursor.execute(create_table_sql)
                if VERBOSE_MODE:
                    print(f"Debug: Created table {table_name}")
            except sqlite3.OperationalError as e:
                print(f"Error creating table {table_name}: {e}")
                continue

        conn.commit()
        
def fetch_all_tags_for_title(table_name, title, verbose):
    global tag_cache_lock
    with sqlite3.connect(args.db, isolation_level=None, check_same_thread=False) as conn:
        cursor = conn.cursor()
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        if isinstance(title, list):
            all_tags = {}
            for single_title in title:
                try:
                    cursor.execute(f"""
                    SELECT tag_key, tag_value, validUntil, id 
                    FROM {table_name} 
                    WHERE Title = ? 
                    AND COALESCE(date(validSince), date('now')-1) <= date('now')
                    AND COALESCE(date(validUntil), '9999-12-31') > date('now')
                    """, (single_title,))
                    
                    tags = {}
                    for tag_key, tag_value, valid_until, id in cursor.fetchall():
                        if valid_until >= current_date:
                            tags[tag_key] = {'value': tag_value, 'validUntil': valid_until, 'id': id}
                            with tag_cache_lock:
                                if single_title not in tag_cache:
                                    tag_cache[single_title] = {}
                                tag_cache[single_title][tag_key] = tags[tag_key]  # Update cache with thread-safety
                    
                    # Merge tags for each title into all_tags
                    all_tags[single_title] = tags
                except sqlite3.Error as e:
                    safe_print(f"Error fetching tags for title {single_title}: {e}", force=True)
            
            return all_tags
        else:
            # If title is not a list, handle it as before
            try:
                cursor.execute(f"""
                SELECT tag_key, tag_value, validUntil, id 
                FROM {table_name} 
                WHERE Title = ? 
                AND COALESCE(date(validSince), date('now')-1) <= date('now')
                AND COALESCE(date(validUntil), '9999-12-31') > date('now')
                """, (title,))
                
                tags = {}
                for tag_key, tag_value, valid_until, id in cursor.fetchall():
                    if valid_until >= current_date:
                        tags[tag_key] = {'value': tag_value, 'validUntil': valid_until, 'id': id}
                        with tag_cache_lock:
                            if title not in tag_cache:
                                tag_cache[title] = {}
                            tag_cache[title][tag_key] = tags[tag_key]  # Update cache with thread-safety
                
                return tags
            except sqlite3.Error as e:
                safe_print(f"Error fetching tags for title {title}: {e}", force=True)
                return {}


def update_or_insert_tag(conn, table_name, record, staging, verbose=False, in_batch=False, existing_tag_values=None):
    global tag_cache

    try:
        i = 0
        u = 0
        sql_return = []
                
        if conn:
            cursor = conn.cursor()
        else:
            return i, u, sql_return
            
        if isinstance(record, dict):
            try:
                new_record_title = record['Title']
                new_record_key = record['tag_key']
                new_record_value = record['tag_value']
                if '_link_' in new_record_key:  
                    new_record_value = urljoin(base_href, new_record_value) if not new_record_value.startswith('http') else new_record_value
            except Exception as e:
                safe_print(f"update_or_insert_tag: Error working with record as a dict: {e}", force=True)
                return i, u, None
        else:
            return i, u, None

        # if DEBUG_MODE:
            # print(f"\t\t- {new_record_key} = {new_record_value}")
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        existing_tag_value = None
        existing_tag_id = None

        # hmmm
        test = False
        
        
        try:
            if not staging:
                # Use cached value if available
                if existing_tag_values is not None: 
                    try:
                        if len(existing_tag_values) > 0:
                            existing_tag_value = existing_tag_values.get(new_record_key, {}).get('value')
                            existing_tag_id = existing_tag_values.get(new_record_key, {}).get('id')  
                        else:
                            existing_tag_value = None
                    except Exception as e:
                        safe_print(f"update_or_insert_tag: Error accessing existing_tag_values: {e}", force=True)

                if existing_tag_value is None and existing_tag_values is not None:
                    try:
                        # cache issues?
                        # print(f"existing_tag_value is None")
                        sql = f"""
                        SELECT tag_value, Id, validUntil 
                        FROM {table_name} 
                        WHERE Title = ? AND tag_key = ? 
                        AND COALESCE(date(validSince), date('now')-1) <= date('now')
                        AND COALESCE(date(validUntil), '9999-12-31') > date('now')
                        ORDER BY COALESCE(date(validSince), CreatedDate) DESC LIMIT 1
                        """
                        
                        formatted_sql = sql.replace("?", "'{}'").format(new_record_title, new_record_key).replace('\n', '').replace('\r', '')
                        if DEBUG_MODE:
                            safe_print(f"update_or_insert_tag: Selecting exisiting value with {formatted_sql}")
                            
                        cursor.execute(sql, (new_record_title, new_record_key))

                        existing_tag = cursor.fetchone()
                        if existing_tag:
                            existing_tag_value, existing_tag_id, existing_validUntil = existing_tag
                            with tag_cache_lock:
                                if new_record_title not in tag_cache:
                                    tag_cache[new_record_title] = {}
                                tag_cache[new_record_title][new_record_key] = {'value': existing_tag_value, 'validUntil': existing_validUntil}
                        else:
                            if DEBUG_MODE:
                                safe_print(f"update_or_insert_tag: No existing tag found for {new_record_key}")
                            existing_tag_value = None
                            existing_tag_id = None 
                            existing_validUntil = None
                                
                    except sqlite3.Error as e:
                        if 'bad parameter or other API misuse' in e:
                            if VERBOSE_MODE:
                                safe_print(f"update_or_insert_tag: Error selecting existing tag using query:\n{formatted_sql}\n{e}", force=True)
                        else:
                            if DEBUG_MODE:
                                safe_print(f"update_or_insert_tag: Error selecting existing tag using query:\n{formatted_sql}\n{e}", force=True)
                        # conn.rollback()
                        # return i, u, None  
                    except Exception as e:
                        safe_print(f"update_or_insert_tag: Error in update_or_insert_tag when existing tag is None, {e}", force=True)

                if existing_tag_value is not None:

                    if not existing_tag_id:
                        if DEBUG_MODE:
                            safe_print(f"update_or_insert_tag: No existing_tag_id with value, try SELECT for {new_record_key}")
                        sql = f"""
                            SELECT tag_value, Id, validUntil 
                            FROM {table_name} 
                            WHERE Title = ? AND tag_key = ? 
                            AND COALESCE(date(validSince), date('now')-1) <= date('now')
                            AND COALESCE(date(validUntil), '9999-12-31') > date('now')
                            ORDER BY COALESCE(date(validSince), CreatedDate) DESC LIMIT 1
                            """
                        cursor.execute(sql, (new_record_title, new_record_key))
                        
                        existing_tag = cursor.fetchone()
                        if existing_tag:
                            existing_tag_value, existing_tag_id, existing_validUntil = existing_tag
                            with tag_cache_lock:
                                if new_record_title not in tag_cache:
                                    tag_cache[new_record_title] = {}
                                tag_cache[new_record_title][new_record_key] = {'value': existing_tag_value, 'validUntil': existing_validUntil}
                        else:
                            if DEBUG_MODE:
                                safe_print(f"update_or_insert_tag: No existing tag found for {new_record_key}")
                        
                    if new_record_value != existing_tag_value and existing_tag_id:
                        if DEBUG_MODE:
                            print(f"New {new_record_value} ain't old {existing_tag_value}")
                        try:
                            sql = f"""
                            UPDATE {table_name}
                            SET validUntil = ?
                            WHERE Id = ?
                            """.strip()
                            
                            formatted_sql = sql.replace("?", "'{}'").format(current_date, existing_tag_id)
                            
                            if verbose:
                                safe_print(f"Executing SQL: {formatted_sql}")

                            if in_batch:
                                sql_return.append(formatted_sql)
                            else:
                                cursor.execute(sql, (current_date, existing_tag_id))
                            
                            u += 1
                                
                        except sqlite3.Error as e:
                            if not in_batch:
                                safe_print(f"update_or_insert_tag: Error updating old tag's validUntil {in_batch}: {e}", force=True)
                                conn.rollback()
                                
                        except Exception as e:
                            safe_print(f"update_or_insert_tag: Error in update_or_insert_tag when updating validUntil, {e}", force=True)
                        
                        try:
                            sql = f"""
                            INSERT INTO {table_name} (Id, Title, tag_key, tag_value, validSince, validUntil, CreatedDate, ModifiedDate) 
                            VALUES (?, ?, ?, ?, ?, '9999-12-31', ?, ?)
                            """.strip()
                            new_id = str(uuid4())
                            
                            formatted_sql = sql.replace("?", "'{}'").format(new_id, new_record_title, new_record_key.replace("'", "''"), new_record_value.replace("'", "''"), current_date, current_date_time, current_date_time).replace('\n', '').replace('\r', '')

                            if in_batch:
                                sql_return.append(formatted_sql)
                                if verbose:
                                    safe_print(f"update_or_insert_tag: existing tag ({existing_tag_value}) has a new value 0: {formatted_sql}")
                            else:
                                cursor.execute(sql, (new_id, new_record_title, new_record_key, new_record_value, current_date, current_date_time, current_date_time))
                                if DEBUG_MODE:
                                    print(f"update_or_insert_tag: Debug: existing tag ({existing_tag_value}) has a new value {new_record_key}: {new_record_value}")
                            
                            i += 1
                            
                            if DEBUG_MODE:
                                safe_print(f"update_or_insert_tag: Debug: Updated and inserted new record for {new_record_key}: {new_record_value}")
                            
                            # Update cache for new tag value
                            with tag_cache_lock:
                                if new_record_title not in tag_cache:
                                    tag_cache[new_record_title] = {}
                                tag_cache[new_record_title][new_record_key] = {'value': new_record_value, 'validUntil': '9999-12-31', 'id': new_id}
                                
                        except sqlite3.Error as e:
                            if not in_batch:
                                safe_print(f"update_or_insert_tag: Error inserting new tag record {in_batch}: {e}", force=True)
                                conn.rollback()
                            return i, u, sql_return

                        except Exception as e:
                            safe_print(f"update_or_insert_tag: Error in update_or_insert_tag when inserting new tag, {e}", force=True)
                            
                    if new_record_value != existing_tag_value and not existing_tag_id:
                        existing_tag_value = None
                        
                    if new_record_value == existing_tag_value and existing_tag_id:
                        try:
                            sql = f"""
                            UPDATE {table_name}
                            SET ModifiedDate = ?
                            WHERE Id = ?
                            """.strip()
                            
                            if existing_tag_id:
                                formatted_sql = sql.replace("?", "'{}'").format(current_date_time, existing_tag_id)
                                if in_batch:
                                    sql_return.append(formatted_sql)
                                else:
                                    cursor.execute(sql, (current_date_time, existing_tag_id))
                            
                            u += 1
                            if DEBUG_MODE:
                                print(f"update_or_insert_tag: Debug: Updated ModifiedDate for unchanged value of {new_record_key}")
                                
                            return i, u, sql_return
                            
                        except sqlite3.Error as e:
                            if not in_batch:
                                safe_print(f"update_or_insert_tag: Error updating ModifiedDate (unchanged value) {in_batch}: {e}", force=True)
                                conn.rollback()
                            return i, u, sql_return
                            
                        except Exception as e:
                            safe_print(f"update_or_insert_tag: Error in update_or_insert_tag when updating tag ModifiedDate, {e}", force=True)
                        
                if existing_tag_value is None:
                    try:
                        sql = f"""
                        INSERT INTO {table_name} (Id, Title, tag_key, tag_value, validSince, validUntil, CreatedDate, ModifiedDate) 
                        VALUES (?, ?, ?, ?, ?, '9999-12-31', ?, ?)
                        """.strip()
                        new_id = str(uuid4())
                        
                        formatted_sql = sql.replace("?", "'{}'").format(new_id, new_record_title, new_record_key.replace("'", "''"), new_record_value.replace("'", "''"), current_date, current_date_time, current_date_time).replace('\n', '').replace('\r', '')
                        
                        if in_batch:
                            sql_return.append(formatted_sql)
                            # if verbose:
                                # safe_print(f"update_or_insert_tag: adding SQL for new tag to batch 1: {formatted_sql}")
                        else:
                            cursor.execute(sql, (new_id, new_record_title, new_record_key, new_record_value, current_date, current_date_time, current_date_time))
                            if DEBUG_MODE:
                                print(f"update_or_insert_tag: Debug: inserted new tag {new_record_key}: {new_record_value}")
                            
                        i += 1
                        # Update cache for new tag
                        with tag_cache_lock:
                            if new_record_title not in tag_cache:
                                tag_cache[new_record_title] = {}
                            tag_cache[new_record_title][new_record_key] = {'value': new_record_value, 'validUntil': '9999-12-31', 'id': new_id}
                            
                    except sqlite3.Error as e:
                        if not in_batch:
                            safe_print(f"update_or_insert_tag: Error inserting brand new tag {in_batch}: {e}")
                            safe_print(f"{new_record_key} : {new_record_value}")
                            conn.rollback()
                            return i, u, None
                    except Exception as e:
                        safe_print(f"update_or_insert_tag: Error in update_or_insert_tag when inserting brand new tag, {e}", force=True)
                    
                # here
                try:
                    if not in_batch:
                        conn.commit()
                except Exception as e:
                    safe_print(f"update_or_insert_tag: error with conn.commit(), {e}")
                    pass
                
                return i, u, sql_return
            elif staging:
                try:
                    sql = f"""
                    INSERT INTO {table_name}_staging (Id, Title, tag_key, tag_value, validSince, validUntil, CreatedDate, ModifiedDate) 
                    VALUES (?, ?, ?, ?, ?, '9999-12-31', ?, ?)
                    """.strip()
                    new_id = str(uuid4())
                    
                    formatted_sql = sql.replace("?", "'{}'").format(new_id, new_record_title, new_record_key.replace("'", "''"), new_record_value.replace("'", "''"), current_date, current_date_time, current_date_time).replace('\n', '').replace('\r', '')
                    
                    sql_return.append(formatted_sql)
                    i += 1
                    
                    if verbose:
                        print(f"update_or_insert_tag: staging: inserted new sql command for {new_record_key}: {new_record_value}")
                    
                    return i, u, sql_return
                
                except Exception as e:
                    safe_print(f"update_or_insert_tag: staging: error inserting sql command tag, {e}", force=True)
                    return i, u, sql_return
                            
        except sqlite3.Error as e:
            if "not an error" in str(e) or "no transaction is active" in str(e):
                safe_print(f"update_or_insert_tag: SQLite transaction issue during commit {in_batch}: {e}", force=True)
                # Here, you might want to check if a transaction is active before retrying or handling differently
            else:
                safe_print(f"update_or_insert_tag: SQLite error in during commit {in_batch}: {e}", force=True)
            if not in_batch:
                conn.rollback()
            return i, u, sql_return
            
        except Exception as e:
            if "argument of type 'InterfaceError' is not iterable" in e:
                pass
            else:
                safe_print(f"update_or_insert_tag: Error in update_or_insert_tag {e}", force=True)
                
            if not in_batch:
                conn.rollback()
            return i, u, sql_return
            
    except UnboundLocalError as e:
        safe_print(f"update_or_insert_tag: UnboundLocalError in update_or_insert_tag: {e}", force=True)
    except Exception as e:
        if existing_tag_values:
            safe_print(f"update_or_insert_tag: Unexpected error in update_or_insert_tag with {len(existing_tag_values)} ex. values: {e}\n{record}", force=True)
    finally:
        if not in_batch:
            try:
                conn.commit()
            except Exception as e:
                safe_print(f"update_or_insert_tag: Failed to commit transaction: {e}", force=True)
    
    return i, u, sql_return
        
def process_records(conn, table_name, records, url, staging, verbose=False, in_batch=False):
    title_tags = {}
    inserted_records = 0
    updated_records = 0
    sql_results = []

    with sqlite3.connect(args.db, isolation_level=None, check_same_thread=False) as cache_conn:
        title_tags[url] = fetch_all_tags_for_title(table_name, url, verbose)
    
    for record in records:
        try:
            # Process each record with the pre-fetched tags
            result = update_or_insert_tag(conn, table_name, record, staging, verbose, in_batch, existing_tag_values=title_tags[url])
            
            if result and not isinstance(result, TypeError):
                i = result[0]
                u = result[1]
                if i > 0 or u > 0:
                    sql = result[2]
                    inserted_records += i
                    updated_records += u
                    sql_results.append(sql)
            else:
                # safe_print(f"TypeError while handling {table_name} with record: {record}\nand existing {title_tags[url]}")
                result = update_or_insert_tag(conn, table_name, record, staging, verbose, in_batch, existing_tag_values=title_tags[url])
                if result and not isinstance(result, TypeError):
                    i = result[0]
                    u = result[1]
                    if i > 0 or u > 0:
                        sql = result[2]
                        inserted_records += i
                        updated_records += u
                        sql_results.append(sql)
                else:
                    safe_print(f"process_records: TypeError (2) while handling {table_name}")
                    safe_print(f"process_records: TypeError (2) Handling {record}")
        
        except TypeError as e:
            safe_print(f"process_records: TypeError in process_records {e}", force=True)
        except Exception as e:
            safe_print(f"process_records: exception error {e}", force=True)
        
    return inserted_records, updated_records, sql_results

def generate_filename_from_url(url):
    """
    Generate a filename from a URL, incorporating pagination parameters if not present.
    Also reconstructs the URL with default pagination parameters if not specified.

    Args:
    url (str): The URL to parse for filename generation.

    Returns:
    tuple: A tuple containing:
        - str: The generated filename based on the URL parameters.
        - str: The base filename.
        - str: The reconstructed URL with all pagination parameters included.
    """
    try:
        
        # Parse the URL
        parsed_url = urlparse(url)
        
        # Extract the base filename from the path (last part before query parameters)
        path_parts = parsed_url.path.split('/')
        base_filename = path_parts[-1] if path_parts else 'unknown_filename'  # Use the last part of the path
        
        # Parse query parameters
        query_params = parse_qs(parsed_url.query)
        
        # List of parameters to look for, in order
        pagination_params = ['pu', 'pa', 'pn', 'po']
        
        # Build the filename components from query parameters
        param_components = []
        for param in pagination_params:
            if param in query_params:
                param_components.extend(f"{param}{value}" for value in query_params[param])
            else:
                # If parameter not in URL, use default value '1'
                query_params[param] = ['1']  # Add this parameter with default '1'
                param_components.append(f"{param}1")
        
        # Construct filename
        if param_components:
            filename = f"{base_filename}_{'_'.join(param_components)}.html"
        else:
            filename = f"{base_filename}.html"
        
        # Reconstruct the URL with all parameters
        new_query = urlencode(query_params, doseq=True)
        reconstructed_url = urlunparse(parsed_url._replace(query=new_query))

    except Exception as e:
        safe_print(f"generate_filename_from_url: an error occured handling {url}: {e}")
        filename = ""
        base_filename = ""
        reconstructed_url = url
        
    return filename, base_filename, reconstructed_url


@retry(stop=stop_after_attempt(1), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_page(url, session_or_driver):
    """Fetch the web page content, first checking for a local copy, then using the given URL with retry logic.
    This function now works with either a requests session or a selenium driver."""
    
    tab_prefix = "\t"
    
    if DEBUG_MODE:
        safe_print(f"fetch_page: session_or_driver type is {type(session_or_driver).__name__}")

    # Get the last part of the URL for the filename
    try:
        is_search_page = url.index("/zoeken?")
    except Exception:
        is_search_page = None

    if is_search_page and is_search_page > -1:
        params = url.split('&')
        page_param = next((param for param in params if param.startswith('page=')), None)
        page_value = page_param.split('=')[1] if page_param else "0"
        base_filename = f"zoeken_page_{page_value}"
        filename = f"{base_filename}.html"
        reconstructed_url = url
    else:
        filename, base_filename, reconstructed_url = generate_filename_from_url(url)

    local_path_pure = os.path.join(".\\local_cache", f"{base_filename}.html")
    local_path = os.path.join(".\\local_cache", f"{filename}")

    # Check if there's a local copy
    if not FORCE_MODE:
        for path in [local_path, local_path_pure]:
            if os.path.exists(path):
                if VERBOSE_MODE:
                    safe_print(f"{tab_prefix}- fetch_page: reading local copy from {path}")
                try:
                    with open(path, 'r', encoding='utf-8') as file:
                        html_content = file.read()
                    return 200, BeautifulSoup(html_content, 'html.parser')
                except IOError as e:
                    safe_print(f"{tab_prefix}- fetch_page: error reading local file at {path}: {e}", force=True)

    # If no local copy or error reading local file or --force, fetch from the URL
    try:
        if isinstance(session_or_driver, webdriver.remote.webdriver.WebDriver):
            # Using Selenium WebDriver
            session_or_driver.get(reconstructed_url)
            WebDriverWait(session_or_driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            soup = BeautifulSoup(session_or_driver.page_source, 'html.parser')
            status_code = 200  # Assuming successful fetch with WebDriver since we can't get HTTP status codes directly
        else:
            # Using requests Session
            if VERBOSE_MODE:
                safe_print(f"{tab_prefix}- fetch_page reconstructed url: {reconstructed_url}")
            response = session_or_driver.get(reconstructed_url, headers=headers, timeout=4)
            response.raise_for_status()
            status_code = response.status_code
            soup = BeautifulSoup(response.text, 'html.parser')

        # Save the content to local cache
        with open(local_path, 'w', encoding='utf-8') as file:
            if DEBUG_MODE:
                safe_print(f"{tab_prefix}- fetch_page: saved local copy {local_path}")
            file.write(str(soup))

        return status_code, soup

    except (TimeoutException, WebDriverException, Exception) as e:
        safe_print(f"{tab_prefix}- fetch_page: exception raised: {e}", force=True)
        return "999-exception error", None

def download_inventaris(url, file_name, session):
    """Download the inventaris file if necessary. Skip if file exists and --force is not used."""
    local_file_path = os.path.join(DOWNLOAD_INVENTARIS, file_name)
    if os.path.exists(local_file_path) and not args.force:
        if VERBOSE_MODE:
            print(f"File {file_name} already exists. Skipping download as --force not used.")
        return local_file_path

    if url:
        try:
            response = session.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            with open(local_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            if VERBOSE_MODE:
                print(f"Successfully downloaded {file_name}")
            return local_file_path
        except requests.RequestException as e:
            if VERBOSE_MODE:
                print(f"Error downloading {url}: {e}")
            return None
    else:
        if VERBOSE_MODE:
            print(f"No URL provided for file {file_name}. Skipping download.")
        return None

def extract_tags_from_ul(soup):
    results = soup.find(id="search-results-list")
    if results:
        for result in results.find_all('li', class_='woo-search-result'):

            header = result.find('header', class_='woo-search-result__header')
            if not header:
                continue

            """Extract tags from the table within an individual search result."""
            tags = {}
            
            title = header.find('h3', class_='woo-search-result__title').text.strip()
            tags[f"title"] = title
            href = header.find('a', class_='woo-search-result__main-link')['href']
            tags[f"dossier_url"] = href


            # First spec list in the header
            first_spec_list = header.find('ul', class_='woo-search-result__spec-list')
            if first_spec_list:
                spec_items = first_spec_list.find_all('li', class_='woo-search-result__spec')
                tags[f"decision_type"] = spec_items[0].find('span', class_='font-bold').text if len(spec_items) > 0 else ""
                tags[f"document_count_text"] = spec_items[1].text.strip() if len(spec_items) > 1 else ""
                tags[f"document_count_ori"] = document_count_text  # Keep the original text
                # Strip non-numeric characters from document_count
                tags[f"document_count"] = ''.join(filter(str.isdigit, document_count_text)) or "0"
                tags["disclosure_type"] = spec_items[2].text.strip() if len(spec_items) > 2 else ""
            # else:
                # tags[f"decision_type"], tags[f"document_count"], tags[f"document_count_ori"], tags[f"disclosure_type"] = "", "0", "", ""

            # Second spec list for dates and dossier number
            second_spec_list = result.find_all('ul', class_='woo-search-result__spec-list')[-1]
            decision_date = None
            publication_date = None
            dossier_number = ""
            if second_spec_list:
                specs = second_spec_list.find_all('li', class_='woo-search-result__spec')
                for spec in specs:
                    if spec.find('time'):
                        time_elements = spec.find_all('time')
                        if time_elements:
                            if 'Besluit genomen op' in spec.text:
                                tags[f"decision_date"] = time_elements[0]['datetime']
                            if 'gepubliceerd op' in spec.text:
                                tags[f"publication_date"] = time_elements[0]['datetime']
                    else:
                        tags[f"dossier_number"] = spec.text.strip()
                    
            table_div = result.find('div', class_='woo-responsive-table')
            if table_div:
                table = table_div.find('table', class_='woo-table')
                if table:
                    for row in table.find_all('tr'):
                        th = row.find('th')
                        td = row.find('td')
                        if th and td:
                            key = th.get_text(strip=True)
                            value = td.get_text(strip=True)
                            link = td.find('a', href=True)
                            if link:
                                tags[f"{key}_link"] = link['href']
                                tags[f"{key}_linkname"] = link.get_text(strip=True)
                            tags[key] = value
            
            if VERBOSE_MODE:
                print("Tags (one result): ", tags)
            
        if VERBOSE_MODE:
            print("Tags (final): ", tags)

        return tags
        
    return {}

def extract_tags_from_table(soup):
    tags = {}
    table_div = soup.find('div', class_='woo-responsive-table')
    if table_div:
        table = table_div.find('table', class_='woo-table')
        if table:
            for row in table.find_all('tr'):
                th = row.find('th')
                td = row.find('td')
                if th and td:
                    key = th.get_text(strip=True)
                    value = td.get_text(strip=True).replace('\n', ' ').replace(' +', ' ')
                    value = re.sub(r'\s+', ' ', value).strip()
                    tags[key] = value
                    links = td.find_all('a', href=True)
                    for i, link in enumerate(links, 1):
                        href = link['href']
                        full_url = href if href.startswith(('http://', 'https://')) else urljoin(base_href, href)
                        tags[f"{key}_link_{i}"] = full_url
                        value = link.get_text(strip=True).replace('\n', ' ').replace(' +', ' ')
                        value = re.sub(r'\s+', ' ', value).strip()
                        tags[f"{key}_linkname_{i}"] = value
    return tags

def extract_history_tags_from_table(soup, url, current_date, current_date_time):
    tags = {}
    tag_data = []
    
    table_div = soup.find('div', attrs={'data-e2e-name': 'document-history'})
    if table_div:
        table = table_div.find('table', class_='woo-table')
        if table:
            headers = [header.text.strip() for header in table.find('thead').find_all('th')]
            for n, row in enumerate(table.find('tbody').find_all('tr'), 1):
                columns = row.find_all('td')
                if len(columns) == len(headers):
                    for header, data in zip(headers, columns):
                        key = f"History {header}_{n}"
                        value = data.get_text(strip=True).replace('\n', ' ').replace(' +', ' ')
                        value = re.sub(r'\s+', ' ', value).strip()
                        # tags[key] = value
                        tag_data.append({
                                        'Title': url,
                                        'tag_key': key,
                                        'tag_value': value,
                                        'validSince': current_date,
                                        'validUntil': '9999-12-31',
                                        'CreatedDate': current_date_time
                                    })
                                    
                        links = data.find_all('a', href=True)
                        if links:
                            for i, link in enumerate(links, 1):
                                href = link['href']
                                full_url = href if href.startswith(('http://', 'https://')) else urljoin(base_href, href)
                                tag_data.append({
                                                'Title': url,
                                                'tag_key': f'History linkhref_{i}',
                                                'tag_value': full_url,
                                                'validSince': current_date,
                                                'validUntil': '9999-12-31',
                                                'CreatedDate': current_date_time
                                            })
                                            
                                # tags[f"{key}_link_{i}"] = full_url
                                value = link.get_text(strip=True).replace('\n', ' ').replace(' +', ' ')
                                value = re.sub(r'\s+', ' ', value).strip()
                                # tags[f"{key}_linkname_{i}"] = value
                                tag_data.append({
                                                'Title': url,
                                                'tag_key': f'History linkname_{i}',
                                                'tag_value': value,
                                                'validSince': current_date,
                                                'validUntil': '9999-12-31',
                                                'CreatedDate': current_date_time
                                            })
    
    return tag_data

def access_elements(data):
    if data is None:
        return None

    # Convert string representation to Python object if necessary
    if isinstance(data, str):
        try:
            data = eval(data)
        except:
            return f"Invalid input format. Please ensure the input is a valid Python list or string representation: {data}."

    # Check if data is a list or tuple with at least two elements
    if not isinstance(data, (list, tuple)) or len(data) < 2:
        return "Data structure does not match expected format; needs at least two elements in a list or tuple."

    # First part: list of dictionaries
    dict_list = data[0]
    if not isinstance(dict_list, list):
        return "First element should be a list of dictionaries."

    # Extract dictionary information, allowing for different numbers of dictionaries
    dict_info = []
    for d in dict_list:
        if isinstance(d, dict):
            dict_info.append(d)
        else:
            return f"Expected dictionary, got {type(d)} in first element."

    list_of_lists = data[1]
    if not isinstance(list_of_lists, list):
        return "Second element should be a list."

    document_info = []
    for item in list_of_lists:
        if isinstance(item, dict) and all(key in item for key in ['document_id', 'document_type', 'document_name', 'document_date', 'document_url']):
            document_info.append(item)
        elif isinstance(item, list) and len(item) >= 5:
            document_info.append({
                'document_id': item[0],
                'document_type': item[1],
                'document_name': item[2],
                'document_date': item[3],
                'document_url': item[4]
            })
        else:
            return f"Unexpected format in second element: {item}"
            
    return dict_info, document_info

def extract_all_tags_from_soup(soup, url, session, current_date, verbose, nodoc, all_documents, documents_set, doc_nr, page_count=None):
    
    pre_text = '- not set- '
    results = []
    total_tags = 0
    sql_results = []
    if page_count is None:
        safe_print("extract_all_tags_from_soup: setting page_count to 1 due to missing page_count")
        page_count = 1
    
    try:
        """Extract all tags from the soup object."""
        tags = []
        
        # we do this from the tp to the bottom
        # div class = 'woo-muted'
        #   Gepubliceerd op
        #   Documentnummer
        # get_div_sections = 'woo-muted'
        
        if not soup:
            result = fetch_page(url, session)
            if result:
                status_code = result[0]
                soup = result[1]
        
        if soup:
            if DEBUG_MODE:
                safe_print(f"extract_all_tags_from_soup: ready to parse page {type(soup)}")
            
            current_date = datetime.now().strftime('%Y-%m-%d')
            current_date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            tabprefix = '\t\t'
            document_references = []
            
            total_tags = 0
            
            ###########################################
            #### vars: download_button, page, tabprefix
            # request
            # https://open.minvws.nl/dossier/VWS-WC/001
            # woo-bordered-button woo-bordered-button--primary
            
            if 'document' not in url.lower():
                # check download button
                download_button = soup.find('button', class_='woo-bordered-button woo-bordered-button--primary', attrs={'data-e2e-name': 'download-documents-button'})
                page = 'request'
                ltable_name = 'requests_tags'
                tabprefix = '\t'
                if not download_button:
                    download_button = soup.find('a', class_='woo-button woo-button--primary', attrs={'data-e2e-name': 'download-file-link'})
            
            elif 'document' in url.lower():
                # document page
                page = 'document'
                tabprefix = '\t\t'
                ltable_name = 'documents_tags'
                download_button = soup.find('a', class_='woo-button woo-button--primary', attrs={'data-e2e-name': 'download-file-link'})
              
            elif 'dossier' in url.lower():
                page = 'request'
                tabprefix = '\t'
                ltable_name = 'requests_tags'
                download_button = soup.find('button', class_='woo-button woo-button--primary', attrs={'data-e2e-name': 'download-file-link'})  
                
            else:
                # something strange
                page = 'unknown'
                tabprefix = ''
                download_button = soup.find('form', class_='sm:flex gap-x-4 items-center')
            
            try:
                publication_date = soup.find('time')['datetime']
                document_number = soup.find('dt', class_='inline').find_next_sibling('dd', class_='inline').text.strip()
                h1 = soup.find('h1')
                # Extract values starting from h1
                filename = h1.find_all('span')[-1].text.strip()
                hidden_text = h1.find('span', class_='sr-only').text.strip()
                tags.append({'Title': url, 'tag_key': f"{page}_documentName", 'tag_value': filename})
                tags.append({'Title': url, 'tag_key': f"{page}_documentType", 'tag_value': hidden_text})
                tags.append({'Title': url, 'tag_key': f"{page}_documentPublicationDate", 'tag_value': publication_date})
                tags.append({'Title': url, 'tag_key': f"{page}_documentNumber", 'tag_value': document_number})
                total_tags += 4
            except Exception as e:
                publication_date = "n/a"
                document_number = "n/a"
                filename = "n/a"
                hidden_text = "n/a"
                pass
            
            if VERBOSE_MODE:
                safe_print(f"extract_all_tags_from_soup: document {filename}, publication_date {publication_date}, document_number {document_number}")
                
            ###########################################
            #### Download information
            #### tags: request pages:
            ####       DossierDownloadLink,
            ####       Download document archief <header>_<nr>
            ####       Download document archief <header>_href_<nr>
            #### tags: document pages:        
            ####       DocumentDownload_linkname
            ####       DocumentDownload_linkhref
            ####       DocumentDownload_proposed_filename
            ####       DocumentDownload_actual_download_url
            if page_count:
                if not download_button:
                    pre_text = "_SectionDownloadButton"
                    tags.append({'Title': url, 'tag_key': f"{page}{pre_text}", 'tag_value': f"0"})
                    total_tags += 1
                    # if VERBOSE_MODE:
                    safe_print(f"{tabprefix}- no download button or reference detected for {page}")
                elif download_button:
                    pre_text = "_SectionDownloadButton"
                    tags.append({'Title': url, 'tag_key': f"{page}{pre_text}", 'tag_value': f"1"})
                    total_tags += 1
                    safe_print(f"{tabprefix}- scraping download section for {page} page -> DocumentDownload+ ")
                    start_tags = total_tags
                      
                    if page == 'request':
                        try:
                            # Find the form and download button directly from the soup
                            form = soup.find('form', class_='sm:flex gap-x-4 items-center')
                            if form:
                                download_button = form.find('button', attrs={'data-e2e-name': 'download-documents-button'})
                                if download_button:
                                    if VERBOSE_MODE:
                                        safe_print(f"{tabprefix}- download-button: Button found")
                                    
                                    # Note: We can't use find_element with BeautifulSoup, but we can simulate the click with Selenium
                                    try:
                                        driver.get(url)  # Navigate back to the original URL if needed
                                        # Wait for the button to become clickable
                                        button = WebDriverWait(driver, 20).until(
                                            EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-e2e-name='download-documents-button']"))
                                        )
                                        if DEBUG_MODE:
                                            safe_print(f"download-button: Clicking")
                                        button.click()
                                        
                                        # Wait for the page to redirect after clicking
                                        WebDriverWait(driver, 20).until(
                                            EC.url_changes(url)  # Wait until the URL changes from the original one
                                        )
                                        
                                        redirected_url = driver.current_url
                                        if VERBOSE_MODE:
                                            safe_print(f"{tabprefix}- redirected to: {redirected_url}")

                                        # Parse the redirected page's content
                                        redirected_content = driver.page_source
                                        result_soup = BeautifulSoup(redirected_content, 'html.parser')

                                        # Continue with parsing the new page content here
                                        woo_tables = result_soup.find_all('table', class_='woo-table')
                                        if woo_tables:
                                            if DEBUG_MODE:
                                                safe_print(f"download-button: Got woo-tables", woo_tables) 
                                        
                                        dossier_download_link = redirected_url
                                        tags.append({'Title': url, 'tag_key': 'DossierDownloadLink', 'tag_value': f"{dossier_download_link}"})
                                        total_tags += 1
                                        if VERBOSE_MODE:
                                            safe_print(f"{tabprefix}- dossier Download Link: {dossier_download_link}")
                                        
                                        for table in woo_tables:
                                            rows = table.find_all('tr')
                                            for row in rows:
                                                th = row.find('th')
                                                if th:  # Assuming the first cell of each row is a header
                                                    header = th.get_text(strip=True)
                                                    for td in row.find_all('td'):
                                                        cell_text = td.get_text(strip=True)
                                                        cell_text = re.sub(r'\s+', ' ', cell_text).strip()
                                                        if cell_text:
                                                            if header == 'Bestand':
                                                                header = "linkname"
                                                                cell_text = cell_text.replace('Zip-bestand','')
                                                            elif header == 'Documenten':
                                                                header = "documenten"
                                                            elif header == 'Verloopt':
                                                                header = "verloopdatum"
                                                            elif header == 'Link':
                                                                match = re.match(r"(\w+)\((\d+\.\d+ \w+)\)", cell_text)
                                                                if match:
                                                                    action, cell_text = match.groups()
                                                                    # print(f"Action: {action}")
                                                                    # print(f"Size: {size}")
                                                                header = "size"
                                                                
                                                            tags.append({'Title': url, 'tag_key': f"Download document archief_{header}_{row.find_all('td').index(td) + 1}", 'tag_value': f"{cell_text}"})
                                                            total_tags += 1
                                                            
                                                        link = td.find('a', href=True)
                                                        if link:
                                                            if not link['href'].startswith('http'):
                                                                full_link = urljoin(base_href, link['href'])
                                                            else:
                                                                full_link = link['href']
                                                            
                                                            tags.append({'Title': url, 'tag_key': f"Download document archief_linkhref_{row.find_all('td').index(td) + 1}", 'tag_value': f"{full_link}"})
                                                            total_tags += 1
                                                            
                                    except TimeoutException:
                                        safe_print("Timeout waiting for the page to load or for the button to be clickable")
                                    except Exception as e:
                                        safe_print(f"An error occurred during button click or page redirection: {e}")
                                else:
                                    if VERBOSE_MODE:
                                        safe_print(f"{tabprefix}- download button not found within the form")
                            else:
                                if VERBOSE_MODE:
                                    safe_print(f"{tabprefix}- form not found in the document")
                                
                        except AttributeError as e:
                            safe_print(f"An AttributeError occurred while parsing HTML: {e}")
                        except Exception as e:
                            safe_print(f"An unexpected error occurred while processing the download button: {e}")
                    # else:
                        # safe_print("No form_action found for this request page")

                    elif page == 'document':
                        # print(download_button)
                        cell_text = download_button.get_text(strip=True)
                        cell_text = re.sub(r'\s+', ' ', cell_text).strip()
                        if cell_text:
                            tags.append({'Title': url, 'tag_key': f"DocumentDownload_linkname", 'tag_value': f"{cell_text}"})
                            total_tags += 1
                        
                        link = soup.find('a', class_='woo-button woo-button--primary', attrs={'data-e2e-name': 'download-file-link'})
                        if link:
                            if not link['href'].startswith('http'):
                                full_link = urljoin(base_href, link['href'])
                            else:
                                full_link = link['href']
                            
                            tags.append({'Title': url, 'tag_key': f"DocumentDownload_linkhref", 'tag_value': f"{full_link}"})
                            total_tags += 1
                            
                            # on the same site
                            if full_link.startswith(base_href):
                                filename, download_url = extract_filename_and_url_from_headers(full_link)
                                
                                if filename:
                                    tags.append({'Title': url, 'tag_key': f"DocumentDownload_proposed_filename", 'tag_value': filename})
                                    total_tags += 1
                                    # In case download_url is different from full_link, update if necessary
                                    if download_url != full_link:
                                        tags.append({'Title': url, 'tag_key': f"DocumentDownload_actual_download_url", 'tag_value': download_url})
                                        total_tags += 1
                            
                        else:
                            safe_print(f"There is something wrong with {link}")
                        
                        # safe_print(f"({total_tags - start_tags} tags)")
                
            
            ###########################################
            #### Judgement results
            #### tags: Notificaties_<nr>,
            ####       Notificaties_<nr>__linkhref_<nr>
            ####       Notificaties_<nr>__linkname_<nr>
            if page_count == 1:
                notification_divs = soup.find_all('div', attrs={'data-e2e-name': 'notifications'})
                
                if notification_divs:
                    pre_text = "_SectionNotifications"
                    tags.append({'Title': url, 'tag_key': f"{page}{pre_text}", 'tag_value': f"{len(notification_divs)}"})
                    total_tags += 1
                    
                    pre_text = 'Notificaties_'
                    for index, notification_div in enumerate(notification_divs, 1):
                        start_tags = total_tags
                        rich_texts = notification_div.find_all('div', class_='woo-rich-text')
                        for m, rich_text in enumerate(rich_texts, 1):
                            key = f"{pre_text}{m}"
                            value = rich_text.get_text(strip=True)
                            value = re.sub(r'\s+', ' ', value).strip()
                            tags.append({'Title': url, 'tag_key': key, 'tag_value': f"{value}"})
                            total_tags += 1
                            links = rich_text.find_all('a', href=True)
                            for i, link in enumerate(links, 1):
                                href = link.get('href')
                                if href:
                                    full_url = href if href.startswith(('http://', 'https://')) else urljoin(base_href, href)
                                    tags.append({'Title': url, 'tag_key': f"{key}_linkhref_{i}", 'tag_value': full_url})
                                    total_tags += 1
                                    value = link.get_text(strip=True).replace('\n', ' ').replace(' +', ' ')
                                    value = re.sub(r'\s+', ' ', value).strip()
                                    tags.append({'Title': url, 'tag_key': f"{key}_linkname_{i}", 'tag_value': f"{value}"})
                                    total_tags += 1
                    safe_print(f"{tabprefix}- scraping notification section -> {pre_text} ({total_tags - start_tags} tags)")

            ###########################################
            #### Summary results
            #### tags: Summary_<nr>,
            ####       Summary_<nr>_linkhref_<nr>
            ####       Summary_<nr>_linkname_<nr>
            if page_count == 1:
                summary_p = soup.find_all('p', attrs={'data-e2e-name': 'dossier-summary'})
                if summary_p:
                    m = 0
                    pre_text = f"_Summary"
                    start_tags = total_tags
                    for summary in summary_p:
                        m += 1
                        
                        tags.append({'Title': url, 'tag_key': f"{page}{pre_text}", 'tag_value': f"{len(summary)}"})
                        total_tags += 1
                        
                        
                        key = f"{page}{pre_text}_{m}"
                        value = summary.get_text(strip=True)
                        value = re.sub(r'\s+', ' ', value).strip()
                        
                        tags.append({'Title': url, 'tag_key': key, 'tag_value': f"{value}"})
                        total_tags += 1
                        
                        links = summary.find_all('a', href=True)
                        for i, link in enumerate(links, 1):
                            href = link.get('href')
                            if href:
                                full_url = href if href.startswith(('http://', 'https://')) else urljoin(base_href, href)
                                tags.append({'Title': url, 'tag_key': f"{key}_linkhref_{i}", 'tag_value': full_url})
                                total_tags += 1
                                value = link.get_text(strip=True).replace('\n', ' ').replace(' +', ' ')
                                value = re.sub(r'\s+', ' ', value).strip()
                                tags.append({'Title': url, 'tag_key': f"{key}_linkname_{i}", 'tag_value': f"{value}"})
                                total_tags += 1
                        safe_print(f"{tabprefix}- scraping summary section -> {pre_text} ({total_tags - start_tags} tags)")
                            
            ##############################################
            #### The div classes 'mt-16' contain all data.
            #### This part analyzes each div and launches appropriate scrapers for every section.
            #### functions: request pages:
            ####            - none
            #### functions: documents pages:
            ####
            #### tags: request pages:
            ####       - Omvang openbaarmaking
            ####       - Omvang openbaarmaking_documenten
            ####       - Omvang openbaarmaking_paginas
            ####       - Omvang openbaarmaking_linkhref_<nr>
            ####       - Omvang openbaarmaking_linkname_<nr>
            ####       - Omvang openbaarmaking_proposed_filename_<nr>
            ####       - Omvang openbaarmaking_actual_download_url_<nr>
            #### tags: document pages:
            ####       - About_Beoordeling
            ####       - About_Datum document
            ####       - About_Download als
            ####       - About_Gelakte gegevens
            ####       - About_Gelakte gegevens_linkhref_
            ####       - About_Gelakte gegevens_linkname_
            ####       - About_Toelichting
            ####       - About_Toelichting_linkhref_
            ####       - About_Toelichting_linkname_
            ####       - About_Type bronbestand
            ####       - About_Uitzonderingsgrond(en)
            ####       - About_Uitzonderingsgrond(en)_linkhref_
            ####       - About_Uitzonderingsgrond(en)_linkname_
            ####       - About_Beoordeling
            ####       - About_Beoordeling
            ####       - About_Beoordeling
            ####       - 
            if page_count == 1:
                if DEBUG_MODE:
                    safe_print(f"{tabprefix}- looking for mt-16 div classes")
                    
            table_divs = soup.find_all('div', class_='mt-16')
            
            if table_divs:
                if page_count == 1:
                    pre_text = "_SectionHTMLDataDivs"
                    tags.append({'Title': url, 'tag_key': f"{page}{pre_text}", 'tag_value': f"{len(table_divs)}"})
                    total_tags += 1
                
                for div in table_divs:
                    try:
                        # get next h2 header
                        pre_text = '- top -'
                        tag_data = {}
                        document_refs = []
                        results = []
                        
                        if div:
                            h2_text = div.find('h2').text.strip()
                            table = div.find('table', class_='woo-table')
                        else:
                            safe_print(f"Apparently this div does not exist... ")
                            continue
                            
                        if page == 'request':
                            if h2_text == "Over dit besluit" and page_count == 1:
                                pre_text = "_SectionAbout"
                                tags.append({'Title': url, 'tag_key': f"{page}{pre_text}", 'tag_value': f"1"})
                                total_tags += 1
                                if table:
                                    if VERBOSE_MODE:
                                        safe_print(f"\t\tscraping 'Over dit besluit'")
                                    rows = table.find_all('tr')
                                    if rows:
                                        for row in rows:
                                            th = row.find('th')
                                            td = row.find('td')
                                            if th and td:
                                                th_text = th.get_text(strip=True)
                                                td_text = td.get_text(strip=False).replace('\n', ' ').replace(' +', ' ')
                                                td_text = re.sub(r'\s+', ' ', td_text).strip()
                                                tags.append({'Title': url, 'tag_key': th_text, 'tag_value': td_text})
                                                total_tags += 1
                                                    
                                                if th_text == "Omvang openbaarmaking":
                                                    # print('th_text',th_text)
                                                    matches = re.findall(r'(\d{1,3}(?:,\d{3})*(?:\.\d{3})*)\s+(?:document(?:en)?|pagina(?:\'s))', td_text)
                                                    # Extract the numbers
                                                    if matches:
                                                        if len(matches) == 2:
                                                            documents, pages = matches
                                                            tags.append({'Title': url, 'tag_key': th_text + '_documenten', 'tag_value': documents})
                                                            total_tags += 1
                                                            tags.append({'Title': url, 'tag_key': th_text + '_paginas', 'tag_value': pages})
                                                            total_tags += 1
                                                        elif len(matches) == 1:
                                                            documents = matches[0]
                                                            tags.append({'Title': url, 'tag_key': th_text + '_documenten', 'tag_value': documents})
                                                            total_tags += 1
                                                
                                                links = td.find_all('a', href=True)
                                                for i, link in enumerate(links, 1):
                                                    href = link.get('href')
                                                    if href:
                                                        full_url = href if href.startswith(('http://', 'https://')) else urljoin(base_href, href)
                                                        tags.append({'Title': url, 'tag_key': f"{th_text}_linkhref_{i}", 'tag_value': full_url})
                                                        total_tags += 1
                                                        value = link.get_text(strip=True).replace('\n', ' ').replace(' +', ' ')
                                                        value = re.sub(r'\s+', ' ', value).strip()
                                                        tags.append({'Title': url, 'tag_key': f"{th_text}_linkname_{i}", 'tag_value': f"{value}"})
                                                        total_tags += 1
                                                        
                                                        # Check if this link is for downloading a file
                                                        try:
                                                            # Use the custom function to extract filename and URL
                                                            filename, download_url = extract_filename_and_url_from_headers(full_url)
                                                            if filename:
                                                                tags.append({'Title': url, 'tag_key': f"{th_text}_proposed_filename_{i}", 'tag_value': filename})
                                                                total_tags += 1
                                                                # In case download_url is different from full_url, update if necessary
                                                                if download_url != full_url:
                                                                    tags.append({'Title': url, 'tag_key': f"{th_text}_actual_download_url_{i}", 'tag_value': download_url})
                                                                    total_tags += 1
                                                        except Exception as e:
                                                            # Handle exceptions gracefully, perhaps log or print error
                                                            if DEBUG_MODE:
                                                                safe_print(f"Error extracting filename for {full_url}: {str(e)}")

                            if h2_text == "Documenten":  # 
                                # this can be a multipage list, handle through extract_and_process_links
                                # extract_and_process_links needs to return tags ...
                                if page_count == 1:
                                    pre_text = "_SectionDocuments"
                                    tags.append({'Title': url, 'tag_key': f"{page}{pre_text}", 'tag_value': f"1"})
                                    total_tags += 1
                                
                                if not nodoc:
                                    try:
                                        msg = f"{tabprefix}- reading {page} 'Documenten' section..."
                                        # documents_div = div.find('div', attrs={'data-e2e-name': 'documents-section'})
                                        # def extract_and_process_links(html_content, url, session, table_name, documents_table, documents_set=None, conn=None, nodoc=False, in_batch=False, staging=False, pagination=True):

                                        # return len(href_list), all_documents, all_tags if all_tags is not None else [], sql_results if sql_results is not None else []
                                        counted_tags = len(tags)
                                        
                                        # def extract_and_process_links(html_content, url, session, table_name, documents_table, documents_set=None, conn=None, nodoc=False, in_batch=False, staging=False, pagination=True):

                                        results = extract_and_process_links(div, url, session, ltable_name, list(all_documents.values()), documents_set, conn, nodoc, True, False, True)
                                        # html_refs, doc, tags
                                        if results is not None:
                                            #result = access_elements(results)
                                            try:
                                                document_refs = results[1]
                                                tag_data = results[2]
                                                sql_result = results[3]
                                                if sql_result:
                                                    sql_results.append(sql_result)
                                                if tag_data:
                                                    # print(f'\ndocument_refs {results[0]}',document_refs)
                                                    # print(f'\ntag_data',tag_data)
                                                    tags.extend(tag_data)
                                                    total_tags += len(tag_data)
                                                    
                                                if document_refs:
                                                    document_references.append(document_refs)
                                                    safe_print(f"{msg} ({len(tag_data)}/{counted_tags} tags/{len(document_refs)-1} docs of {len(document_refs)-1} total)")
                                                elif not document_refs and tag_data:
                                                    safe_print(f"{msg} ({len(tag_data)}/{counted_tags} tags)")
                                                elif document_refs and not tag_data:
                                                    safe_print(f"{msg} ({len(document_refs)-1} docs, tag issue!)")
                                                    
                                            except Exception as e:
                                                safe_print(f"\nextract_all_tags_from_soup: error reading request documents, {e}")
                                        else:
                                            safe_print(f"\nextract_all_tags_from_soup: extract_and_process_links returned None for request documents")
                                        
                                    except Exception as e:
                                        safe_print(f"\nextract_all_tags_from_soup: an exception occured, {e}")
                                else:
                                    safe_print(f"{tabprefix}- skipping 'Documenten' section because of --nodoc switch")

                        if page == 'document':
                            if h2_text == "Over dit document":  # pre_text = 'About_'
                                pre_text = "_SectionAbout"
                                tags.append({'Title': url, 'tag_key': f"{page}{pre_text}", 'tag_value': f"1"})
                                total_tags += 1
                                
                                pre_text = 'About_'
                                if div:
                                    
                                    # def getwooresponsivetable(tab_div, documents_div, current_url, pre_text, current_date, current_date_time):
                                    # return tag_data, document_references
                                    
                                    # if VERBOSE_MODE:
                                        # print(f'Resulted div {pre_text}: ', div)
                                    # return tag_data, document_references, docs_found
                                    results = getwooresponsivetable(div, "", url, pre_text, current_date, current_date_time)
                                    
                                    if results:
                                        if not results is None:
                                            # print(results)
                                            try:
                                                if len(results) >= 2:
                                                    tag_data = results[0]
                                                    if tag_data and len(tag_data) > 0:
                                                        tags.extend(tag_data)
                                                        total_tags += len(tag_data)
                                                    document_refs = results[1]
                                                    if document_refs and len(document_refs) > 0 and tag_data:
                                                        document_references = document_references.append(document_refs)
                                                        msg = f" ({len(tag_data)} tags/{len(document_references)} docs)"
                                                    elif not document_refs and len(tag_data) > 0:
                                                        msg = f" ({len(tag_data)} tags)"
                                                    elif document_refs and len(document_refs) > 0 and not tag_data:
                                                        msg = f" \n({document_refs} docs, tag issue!)"
                                                    safe_print(f"{tabprefix}- scraping 'Over dit document' section -> {pre_text} {msg}")
                                                else:
                                                    safe_print(f"{tabprefix}- scraping 'Over dit document' section -> {pre_text} ({len(document_refs)} docs, tag issue!)")
                                                    
                                            except Exception as e:
                                                safe_print(f"\nextract_all_tags_from_soup: error reading {pre_text}, {e}\n{results}")
                                        else:
                                            safe_print(f"{tabprefix}- scraping 'Over dit document' section -> {pre_text} ... no results.")
                                        
                                else:
                                    # table = div.find('table', class_='woo-table')
                                    if table:
                                        # if verbose:
                                        safe_print(f"{tabprefix}- scraping 1 'Over dit document' section -> About_")
                                        rows = table.find_all('tr')
                                        if rows and len(rows) > 0:
                                            for row in rows:
                                                th = row.find('th')
                                                td = row.find('td')
                                                if th and td:
                                                    th_text = th.get_text(strip=True)
                                                    td_text = td.get_text(strip=False).replace('\n', ' ').replace(' +', ' ')
                                                    td_text = re.sub(r'\s+', ' ', td_text).strip()
                                                    tags.append({'Title': url, 'tag_key': 'About_' + th_text, 'tag_value': td_text})
                                                    total_tags += 1
                                                    links = td.find_all('a', href=True)
                                                    for i, link in enumerate(links, 1):
                                                        href = link.get('href')
                                                        if href:
                                                            full_url = href if href.startswith(('http://', 'https://')) else urljoin(base_href, href)
                                                            tags.append({'Title': url, 'tag_key': f"About_{th_text}_linkhref_{i}", 'tag_value': full_url})
                                                            total_tags += 1
                                                            value = link.get_text(strip=True).replace('\n', ' ').replace(' +', ' ')
                                                            value = re.sub(r'\s+', ' ', value).strip()
                                                            tags.append({'Title': url, 'tag_key': f"About_{th_text}_linkname_{i}", 'tag_value': f"{value}"})
                                                            total_tags += 1
                                        else:
                                            safe_print(f"{tabprefix}\t- cannot find any rows in table for document {url}?")


                            elif h2_text == "Berichten uit dezelfde e-mailconversatie": # pre_text = "Linked_"
                                pre_text = "_SectionLinked"
                                tags.append({'Title': url, 'tag_key': f"{page}{pre_text}", 'tag_value': f"1"})
                                total_tags += 1
                                
                                if not nodoc:
                                    pre_text = "Linked_"
                                    if div:
                                        documents_div = div.find('div', attrs={'data-e2e-name': 'documents-section'})
                                        if documents_div:
                                            table = documents_div.find('table', class_='woo-table')
                                            if table:
                                                lhref_count = 0
                                                lnumber_inserted = 0
                                                lsql_return = []
                                                
                                                results = extract_and_process_links(documents_div, url, session, ltable_name, list(all_documents.values()), documents_set, conn, nodoc, True)
                                                if results is not None:
                                                    result = access_elements(results)
                                                    if result is not None:
                                                        tag_data, document_refs = result
                                                        print(f"{tabprefix}- scraping 'Berichten uit dezelfde e-mailconversatie' -> {pre_text}", end='')
                                                        try:
                                                            if tag_data:
                                                                tags.extend(tag_data)
                                                                total_tags += len(tag_data)
                                                                
                                                            if document_refs and tag_data:
                                                                document_references = document_references.append(document_refs)
                                                                safe_print(f" ({len(tag_data)} tags/{len(document_references)} docs)")
                                                            elif not document_refs and tag_data:
                                                                safe_print(f" ({len(tag_data)} tags)")
                                                            elif document_refs and not tag_data:
                                                                safe_print(f" ({len(document_refs)} docs, tag issue!)")
                                                                
                                                        except Exception as e:
                                                            safe_print(f"\nextract_all_tags_from_soup: error reading {pre_text}, {e}\nResult: {result}")
                                                    else:
                                                        safe_print(f"\nextract_all_tags_from_soup: access_elements returned None for {pre_text}")
                                                else:
                                                    safe_print(f"\nextract_all_tags_from_soup: extract_and_process_links returned None for {pre_text}")
                                else:
                                    safe_print(f"{tabprefix}- skipping 'Berichten uit dezelfde e-mailconversatie' section because of --nodoc switch")
                                        
                            elif h2_text == "Bijlagen bij dit e-mailbericht":   # pre_text = 'Attachments_'
                                pre_text = "_SectionAttachments"
                                tags.append({'Title': url, 'tag_key': f"{page}{pre_text}", 'tag_value': f"1"})
                                total_tags += 1
                                
                                if not nodoc:
                                    pre_text = 'Attachments_'
                                    if div:
                                        documents_div = div.find('div', attrs={'data-e2e-name': 'documents-section'})
                                        if documents_div:
                                            table = documents_div.find('table', class_='woo-table')
                                            if table:
                                                
                                                lhref_count = 0
                                                lnumber_inserted = 0
                                                lsql_return = []

                                                results = extract_and_process_links(documents_div, url, session, ltable_name, list(all_documents.values()), documents_set, conn, nodoc, True)
                                                
                                                if results is not None:
                                                    result = access_elements(results)
                                                    if result is not None:
                                                        tag_data, document_refs = result
                                                        print(f"{tabprefix}- scraping 'Bijlagen bij dit e-mailbericht' -> {pre_text}", end='')
                                                        try:
                                                            if tag_data:
                                                                tags.extend(tag_data)
                                                                total_tags += len(tag_data)
                                                                
                                                            if document_refs and tag_data:
                                                                document_references = document_references.append(document_refs)
                                                                safe_print(f" ({len(tag_data)} tags/{len(document_references)} docs)")
                                                            elif not document_refs and tag_data:
                                                                safe_print(f" ({len(tag_data)} tags)")
                                                            elif document_refs and not tag_data:
                                                                safe_print(f" ({len(document_refs)} docs, tag issue!)")
                                                                
                                                        except Exception as e:
                                                            safe_print(f"\nextract_all_tags_from_soup: error reading {pre_text}, {e}\nResult: {result}")
                                                    else:
                                                        safe_print(f"\nextract_all_tags_from_soup: access_elements returned None for {pre_text}")
                                                else:
                                                    safe_print(f"\nextract_all_tags_from_soup: extract_and_process_links returned None for {pre_text}")
                                else:
                                    safe_print(f"{tabprefix}- skipping 'Bijlagen bij dit e-mailbericht' section because of --nodoc switch")
                                
                            elif h2_text == "Gerelateerde documenten":      # pre_text = 'Related_'
                                pre_text = "_SectionRelated"
                                tags.append({'Title': url, 'tag_key': f"{page}{pre_text}", 'tag_value': f"1"})
                                total_tags += 1
                                
                                if not nodoc:
                                    pre_text = 'Related_'
                                    if div:
                                        documents_div = div.find('div', attrs={'data-e2e-name': 'documents-section'})
                                        if documents_div:
                                            table = documents_div.find('table', class_='woo-table')
                                    
                                            if table:
                                                lhref_count = 0
                                                lnumber_inserted = 0
                                                lsql_return = []

                                                results = extract_and_process_links(documents_div, url, session, ltable_name, list(all_documents.values()), documents_set, conn, nodoc, True)
                                                # results is 
                                                
                                                if results is not None:
                                                    result = access_elements(results)
                                                    if result is not None:
                                                        tag_data, document_refs = result
                                                        print(f"{tabprefix}- scraping 'Gerelateerde documenten' -> {pre_text}", end='')
                                                        try:
                                                            if tag_data:
                                                                tags.extend(tag_data)
                                                                total_tags += len(tag_data)
                                                                
                                                            if document_refs and tag_data:
                                                                document_references = document_references.append(document_refs)
                                                                safe_print(f" ({len(tag_data)} tags/{len(document_references)} docs)")
                                                            elif not document_refs and tag_data:
                                                                safe_print(f" ({len(tag_data)} tags)")
                                                            elif document_refs and not tag_data:
                                                                safe_print(f" ({len(document_refs)} docs, tag issue!)")
                                                                
                                                        except Exception as e:
                                                            safe_print(f"\nextract_all_tags_from_soup: error reading {pre_text}, {e}\nResult: {result}")
                                                    else:
                                                        safe_print(f"\nextract_all_tags_from_soup: access_elements returned None for {pre_text}")
                                                else:
                                                    safe_print(f"\nextract_all_tags_from_soup: extract_and_process_links returned None for {pre_text}")
                                else:
                                    safe_print(f"{tabprefix}- skipping 'Gerelateerde documenten' section because of --nodoc switch")
                                        
                            elif h2_text == "Documenten die naar dit document verwijzen":   # pre_text = 'Pointed_'
                                pre_text = "_SectionPointed"
                                tags.append({'Title': url, 'tag_key': f"{page}{pre_text}", 'tag_value': f"1"})
                                total_tags += 1
                                
                                if not nodoc:
                                    pre_text = 'Pointed_'
                                    if div:
                                        documents_div = div.find('div', attrs={'data-e2e-name': 'documents-section'})
                                        if documents_div:
                                            table = documents_div.find('table', class_='woo-table')
                                    
                                            if table:
                                                lhref_count = 0
                                                lnumber_inserted = 0
                                                lsql_return = []

                                                results = extract_and_process_links(documents_div, url, session, ltable_name, list(all_documents.values()), documents_set, conn, nodoc, True)
                                                # results is 
                                                
                                                result = access_elements(results)
                                                
                                                if results is not None:
                                                    result = access_elements(results)
                                                    if result is not None:
                                                        tag_data, document_refs = result
                                                        print(f"{tabprefix}- scraping 'Documenten die naar dit document verwijzen' -> {pre_text}", end='')
                                                        try:
                                                            if tag_data:
                                                                tags.extend(tag_data)
                                                                total_tags += len(tag_data)
                                                                
                                                            if document_refs and tag_data:
                                                                document_references = document_references.append(document_refs)
                                                                safe_print(f" ({len(tag_data)} tags/{len(document_references)} docs)")
                                                            elif not document_refs and tag_data:
                                                                safe_print(f" ({len(tag_data)} tags)")
                                                            elif document_refs and not tag_data:
                                                                safe_print(f" ({len(document_refs)} docs, tag issue!)")
                                                                
                                                        except Exception as e:
                                                            safe_print(f"\nextract_all_tags_from_soup: error reading {pre_text}, {e}\nResult: {result}")
                                                    else:
                                                        safe_print(f"\nextract_all_tags_from_soup: access_elements returned None for {pre_text}")
                                                else:
                                                    safe_print(f"\nextract_all_tags_from_soup: extract_and_process_links returned None for {pre_text}")
                                else:
                                    safe_print(f"{tabprefix}- skipping 'Documenten die naar dit document verwijzen' section because of --nodoc switch")

                            elif h2_text == "Achtergrond":      # pre_text = 'Request_'
                                pre_text = "_SectionRequest"
                                tags.append({'Title': url, 'tag_key': f"{page}{pre_text}", 'tag_value': f"1"})
                                total_tags += 1
                                pre_text = 'Request_'
                                if div:
                                    # table = div.find('table', class_='woo-table')
                                    # return tag_data, document_references, total_documents
                                    results = getwooresponsivetable(div, "", url, pre_text, current_date, current_date_time)
                                    
                                    if results:
                                        if not results is None:
                                            
                                            # print(results)
                                            try:
                                                if len(results) >= 2:
                                                    tag_data = results[0]
                                                    if tag_data:
                                                        tags.extend(tag_data)
                                                        total_tags += len(tag_data)
                                                    document_refs = results[1]
                                                    if document_refs and tag_data:
                                                        document_references.append(document_refs)
                                                        msg = f" ({len(tag_data)} tags/{len(document_refs)} and {len(document_references)} docs total)"
                                                    elif not document_refs and tag_data:
                                                        msg = f" ({len(tag_data)} tags)"
                                                    elif document_refs and not tag_data:
                                                        msg = f" ({len(document_refs)} docs, tag issue!)"
                                                    safe_print(f"{tabprefix}- scraping 'Achtergrond' section -> {pre_text} {msg}")
                                                    
                                                else:
                                                    safe_print(f"{tabprefix}- scraping 'Achtergrond' section -> ({len(document_refs)} docs, tag issue!)")
                                                    
                                            except Exception as e:
                                                safe_print(f"\nextract_all_tags_from_soup: error reading woo-table, {e}\n{results}")
                                        else:
                                            safe_print("{tabprefix}- scraping 'Achtergrond' section -> ... no results.")
                                            
                                else:
                                    table = div.find('table', class_='woo-table')
                                    pre_text = 'Request_'
                                    if table:
                                        # if verbose:
                                        safe_print(f"\t\t- scraping 1 'Achtergrond' section -> Request_")
                                        rows = table.find_all('tr')
                                        if rows:
                                            for row in rows:
                                                th = row.find('th')
                                                td = row.find('td')
                                                if th and td:
                                                    th_text = th.get_text(strip=True)
                                                    td_text = td.get_text(strip=False).replace('\n', ' ').replace(' +', ' ')
                                                    td_text = re.sub(r'\s+', ' ', td_text).strip()
                                                    tags.append({'Title': url, 'tag_key': '{pre_text}' + th_text, 'tag_value': td_text})
                                                    total_tags += 1
                                                    
                                                    links = td.find_all('a', href=True)
                                                    for i, link in enumerate(links, 1):
                                                        href = link.get('href')
                                                        if href:
                                                            full_url = href if href.startswith(('http://', 'https://')) else urljoin(base_href, href)
                                                            tags.append({'Title': url, 'tag_key': f"{pre_text}{th_text}_linkhref_{i}", 'tag_value': full_url})
                                                            total_tags += 1
                                                            value = link.get_text(strip=True).replace('\n', ' ').replace(' +', ' ')
                                                            value = re.sub(r'\s+', ' ', value).strip()
                                                            tags.append({'Title': url, 'tag_key': f"{pre_text}{th_text}_linkname_{i}", 'tag_value': f"{value}"})
                                                            total_tags += 1


                            elif h2_text == "Geschiedenis van dit document":    # pre_text = 'History'
                                pre_text = "_SectionHistory"
                                tags.append({'Title': url, 'tag_key': f"{page}{pre_text}", 'tag_value': f"1"})
                                total_tags += 1
                                
                                pre_text = 'History'
                                table = div.find('table', class_='woo-table')
                                
                                if table:
                                    # if verbose:
                                    history_tags = extract_history_tags_from_table(soup, url, current_date, current_date_time)
                                    if history_tags:
                                        safe_print(f"\t\t- scraping 'Geschiedenis van dit document' -> {pre_text} ({int(len(history_tags)/2)} records with {len(history_tags)} tags)")
                                        tags.extend(history_tags)
                                        total_tags += len(history_tags)
                                    else:
                                        print(f"")
                                    # tags.append({'Title': url, 'tag_key': f"{key}_linkname_{i}", 'tag_value': f"{value}"})
                                        
                            else:
                                print(f"\t\t- first pass, skipping header '{h2_text}'")
                    except Exception as e:
                        safe_print(f"extract_all_tags_from_soup: an exception occured while evaluating div {div}, {e}")

        # if total_tags > 0:
            # safe_print(f"{tabprefix}- in total {total_tags} tags scraped")
        
    except AttributeError as e:
        safe_print(f"extract_all_tags_from_soup: error reading, {e}")
        return [], [], []  # Return empty lists or handle accordingly
        
    except Exception as e:
        safe_print(f"extract_all_tags_from_soup: an unexpected error occurred, {e}")
        return [], [], []

    return tags, document_references, sql_results


# Global variables with lock for thread safety
unique_number = 0
doc_href_total = 0
doc_href_inserted = 0
doc_href_updated = 0
number_lock = threading.Lock()

def process_single_document(conn, url, session, verbose, all_documents, documents_set, doc_nr, in_batch=False, nodoc=False):
    global unique_number

    current_date = datetime.now().strftime('%Y-%m-%d')
    with number_lock:
        unique_number += 1

    attempts = 0
    max_attempts = 2
    data = []
    tags = []
    documents = []
    sql_return = []
    
    while attempts < max_attempts:
        sql = ""
        gotError = False
        
        try:
            if DEBUG_MODE:
                safe_print(f"process_single_document: into while attempts {attempts} < max_attempts for {url} and with {session}:", force=True)
            
            safe_print(f"\n- {unique_number:0>7} doc {url}")
            if not url.startswith('URL'):
                result = fetch_page(url, session)
            
            if VERBOSE_MODE and result:
                safe_print(f"process_single_document: fetched page for {url}", force=True)
                
            if result:
                return_status =  result[0]
                soup = result[1]
                # safe_print(f"\n- {unique_number:0>7} doc ({return_status})\t{url}")
                if DEBUG_MODE:
                    safe_print(f"({return_status})\t{url}")
                # def extract_all_tags_from_soup(soup, url, session, current_date, verbose, nodoc, all_documents, documents_set, doc_nr, page_count=None):
                
                if soup:
                    data = extract_all_tags_from_soup(soup, url, session, current_date, verbose, nodoc, all_documents, documents_set, doc_nr, 1)

                if data:
                    tags = data[0]
                    documents = data[1] # already checked against database
                    sql_return.append(data[2])                    
                
                try:
                    # should always be an About_
                    if not isinstance(data, list):
                        datum_document = next((item for item in tags if item.get('tag_key') == 'About_Datum document'), None)
                        beoordeling = next((item for item in tags if item.get('tag_key') == 'About_Beoordeling'), None)
                    else:
                        datum_document = None
                        beoordeling = None
                        
                except Exception as e:
                    safe_print(f"retrieving tag_keys raised an exception: {e}")
                    safe_print(f"{tags}")
                    break
                
                if datum_document and beoordeling:
                    if DEBUG_MODE:
                        safe_print(f"\t\t- {len(tags)} tags, {beoordeling['tag_value']}, datum: {datum_document['tag_value']}")
                elif datum_document and not beoordeling:
                    if DEBUG_MODE:
                        safe_print(f"\t\t- {len(tags)} tags, onbekend, datum: {datum_document['tag_value']}")
                
                return tags, sql_return, documents
            else:
                safe_print(f"- {doc_nr} failed to load ...")
                
        except (requests.HTTPError, requests.RequestException) as errh:
            safe_print(f"Attempt {attempts + 1}/{max_attempts} failed with HTTP Error for {url}: {errh}", verbose)
            # [Locatie open.minvws.nl] = key
            # DocumentStatusCheckedDate = current datetime
            # DocumentStatusCheckedResult = errh
            sql = f"UPDATE documents SET DocumentStatusCheckedDate = CURRENT_TIMESTAMP, DocumentStatusCheckedResult = '{errh}', ModifiedDate = CURRENT_TIMESTAMP WHERE [Locatie open.minvws.nl] = '{url}'"
            gotError = True
            break
            # sleep(random.uniform(1, 3))  # Wait before retrying with backoff
            
        except NameError as e:
            safe_print(f"Attempt {attempts + 1}/{max_attempts} failed with NameError processing {url}: {e}", verbose)
            sql = f"UPDATE documents SET DocumentStatusCheckedDate = CURRENT_TIMESTAMP, DocumentStatusCheckedResult = '{e}', ModifiedDate = CURRENT_TIMESTAMP WHERE [Locatie open.minvws.nl] = '{url}'"
            gotError = True
            break
            
        except Exception as e:
            safe_print(f"Attempt {attempts + 1}/{max_attempts} failed with Unexpected error processing {url}: {e}", verbose)
            sql = f"UPDATE documents SET DocumentStatusCheckedDate = CURRENT_TIMESTAMP, DocumentStatusCheckedResult = '{e}', ModifiedDate = CURRENT_TIMESTAMP WHERE [Locatie open.minvws.nl] = '{url}'"
            gotError = True
            break
            
        attempts += 1
    
    if gotError:
        # safe_print(f"Saving http error status to database for {url}")
        # safe_print("Query", sql)
        try:
            conn.execute(sql)
            conn.commit()
            gotError = False
        except Exception as e:
            safe_print('An error occured saving http status to url,', e)
        
    # safe_print(f"\t\t- gathered {len(data)} tags from single document after {max_attempts} attempt(s)   ", True)
    return tags, sql_return, documents

def process_doc_batch(batch, session, conn, staging, verbose, db_fields, all_documents, documents_set):
    global unique_number, start_time
    
    with session:
    
        # if not isinstance(conn, sqlite3.Connection):
            # conn = create_connection(args.db, TABLE_CONFIGS[0].get('db_platform', 'sqlite3'))
        
        updated_recs = 0
        inserted_recs = 0
        sql_result = []
        result = None
        start_time_batch = datetime.now()
        tabprefix = '\t\t'
        
        if VERBOSE_MODE:
            safe_print(f"Start process_doc_batch with {len(batch)} records, a connection of type {type(conn)}, a list of registered all_documents of {len(all_documents)} and a documents_set of {len(documents_set)}")
        
        for doc in batch:
            error_occured = False
            in_batch = True
            if len(doc) != 2:
                if VERBOSE_MODE:
                    safe_print(f"Skipping document due to unexpected structure: {doc}")
                continue
            try:
                # get information
                if VERBOSE_MODE:
                    safe_print(f"process_doc_batch: into for doc in batch: {doc}")
                
                url, document_id = doc
                if not url.startswith('URL'):
                    tags, sql_result_reply, documents_reply = process_single_document(conn, url, session, verbose, all_documents, documents_set, True)
                    if not tags is None:
                        if len(tags) == 0:
                            continue
                            
                        if DEBUG_MODE:
                            safe_print(f"sql_result_reply: {len(sql_result_reply)} with {len(tags)} tags and {len(documents_reply)} docs")                
                    else:
                        continue
                else:
                    continue
                    
                # if sql_result_reply and not sql_result_reply[0] == "":
                    # for element in sql_result_reply:
                        # if isinstance(element, dict):  # If it's a dictionary
                            # for key, value in element.items():
                                # print(f"Dict: {key} -> {value}")
                        # elif isinstance(element, list):  # If it's another list
                            # # This further nested list contains dictionaries
                            # for sub_element in element:
                                # for key, value in sub_element.items():
                                    # print(f"Nested List Dict: {key} -> {value}")
                                    
            except Exception as e:
                safe_print(f"When launching process_single_document the following error was raised: {e}", force=True)
                error_occured = True
            
            try:
                # new code for batch processing tags per doc
                if VERBOSE_MODE:
                    safe_print(f"Launching process_records(conn, 'documents_tags', {len(tags)}, {verbose}, {in_batch})", force=True)
                
                # return inserted_records, updated_records, sql_results
                # also checks 
                result = process_records(conn, 'documents_tags', tags, url, staging, verbose, in_batch)
                
                if result and len(result) == 3 and (result[0] > 0 or result[1] > 0):
                    inserted_recs += result[0]
                    updated_recs += result[1]
                    if VERBOSE_MODE:
                        safe_print(f"{len(result[2])} is being added to {len(sql_result)}", force=True)
                    sql_result.extend(result[2])
                    count_result = len(result[2])
                    if DEBUG_MODE:
                        safe_print(f"process_doc_batch: count_result: {count_result} sql_result: {len(sql_result)}")
                        
                    if DEBUG_MODE:
                        safe_print(f"Returning results from new process_doc_batch with {len(sql_result)} (nested) results\n{sql_result}", True)
                    continue
                else:
                    # no results
                    if VERBOSE_MODE:
                        safe_print(f"The doc result in process_doc_batch is empty for {url}")
                    
            except Exception as e:
                safe_print(f"When launching process_records the following error was raised: {e}", force=True)
                error_occured = True
                return inserted_recs, updated_recs, sql_result
            
            if not error_occured:
                
                tempblock = False
                if tempblock:
                    for tag in tags:
                        with conn:
                        
                            # Check if tag is a string representation of a list of dictionaries
                            if isinstance(tag, str) and tag.startswith('[') and tag.endswith(']'):
                                import ast
                                tags = ast.literal_eval(tag)  # Safely evaluate the string to a list of dicts
                                if VERBOSE_MODE:
                                    safe_print(f"Handling dictionairy of tags")
                                for single_tag in tags:
                                    if DEBUG_MODE:
                                        safe_print(f"Handling single tag a. {single_tag}")
                                    if isinstance(single_tag, dict):
                                        result = update_or_insert_tag(conn, 'documents_tags', single_tag, staging, verbose, in_batch)
                                    else:
                                        safe_print(f"Warning: Unexpected type for single_tag: {type(single_tag)} with value: {tag}")
                                    
                                    if result and len(result) == 3 and (result[0] > 0 or result[1] > 0):
                                        inserted_recs += result[0]
                                        updated_recs += result[1]
                                        sql_result.extend(result[2])
                                        if VERBOSE_MODE:
                                            safe_print(f"Successfully added 1 {tag}\n{result}")
                                        
                                        result = None
                        
                            elif isinstance(tag, dict):
                                if DEBUG_MODE:
                                    safe_print(f"{tabprefix}- handling single tag b. {tag['tag_key']}")
                                result = update_or_insert_tag(conn, 'documents_tags', tag, staging, verbose, in_batch)
                                if result and len(result) == 3 and (result[0] > 0 or result[1] > 0):
                                    inserted_recs += result[0]
                                    updated_recs += result[1]
                                    sql_result.extend(result[2])
                                    
                                    if VERBOSE_MODE:
                                        safe_print(f"{tabprefix}- Successfully added 2 ({result[0]}/{result[1]})")
                                
                            elif isinstance(tag, list):
                                if not tag == 0 and not tag == '' and not tag == 2:
                                    for single_tag in tag:
                                        if DEBUG_MODE:
                                            safe_print(f"Handling single tag c. {single_tag}")
                                        result = update_or_insert_tag(conn, 'documents_tags', single_tag, staging, verbose, in_batch)
                                        
                                        if DEBUG_MODE:
                                            safe_print(f"Successfully added 3 {single_tag}\n{result}")
                                        
                                        if result and len(result) == 3 and (result[0] > 0 or result[1] > 0):
                                            inserted_recs += result[0]
                                            updated_recs += result[1]
                                            sql_result.extend(result[2])
                            else:
                                safe_print(f"Warning: Unexpected type for tag: {type(tag)} with value: {tag}")
                                
                        result = None

        if VERBOSE_MODE:
            safe_print(f"Returning results from process_doc_batch with {len(sql_result)} results")
            
        return inserted_recs, updated_recs, sql_result

def calculate_batches(total_docs, batch_size):
    # Calculate the number of batches, using ceiling division to ensure all documents are included
    import math
    return math.ceil(total_docs / batch_size)

import concurrent.futures
import time
from datetime import datetime

# ############################################################# #
# Evaluate registered documents against current website status. #
# ----------------------------------------------------- #
# Returns None                                          #
# functions used:
#   - safe_print, calculate_batches, process_doc_batch
# ##################################################### #
# caching headless browser
def evaluate_documents(conn, force, db_fields, query, batch_size, workers, staging, wait_time=10, verbose=False):
    global unique_number, doc_href_total, doc_href_inserted, big_total, start_batch_time, tag_cache
    
    cursor = conn.cursor()
    
    if verbose:
        safe_print(f"Evaluate_documents uses query {query}")
    
    if tag_cache is None:
        tag_cache = {}
        
    retry_execute(cursor, query)
    unique_docs = cursor.fetchall()
    safe_print(f"Unique documents returned from query: {len(unique_docs)}")
    
    results = return_all_registered_documents(conn)
    all_documents = results[0]
    documents_set = results[1]
    
    if VERBOSE_MODE:
        print(f"\t\t- registered documents {len(all_documents)}")
    
    # documents_set = set(doc[0] for doc in all_documents.values())
    
    # all_documents = {doc[0]: doc for doc in unique_docs}

    unique_number = 0
    doc_href_total = 0
    doc_href_inserted = 0
    inserted_recs = 0
    updated_recs = 0
    sql_return = []
    batch_sql_queries = []
    batch_size = int(batch_size)
    big_total = 0
    batch_number = 0
    workers = int(workers)
    
    script_full_path = sys.argv[0]
    script_name, _ = os.path.splitext(script_full_path)
    stop_file = f"{script_name}.stop"
    
    if os.path.exists(stop_file) and not staging:
        stopped = True
        safe_print(f"Stop file '{stop_file}' detected. Stopping the evaluation.", force=True)
        return
    elif os.path.exists(stop_file) and staging:
        stopped = False
        safe_print(f"Stop file '{stop_file}' detected but in staging, continuing.", force=True)
    
    # how many unique_docs (urls) to evaluate
    total_unique_docs = len(unique_docs)
    # how many batches in unique_docs
    number_of_batches = calculate_batches(total_unique_docs, batch_size)
                
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor: 
            with requests.Session() as session:
                if verbose:
                    safe_print(f"evaluate_documents: into with requests.Session() as session:")
                batches = [unique_docs[i:i + batch_size] for i in range(0, len(unique_docs), batch_size)]
                futures = [executor.submit(process_doc_batch, batch, session, conn, staging, verbose, db_fields, all_documents, documents_set) for batch in batches]
                
                results = []
                # Counter for records processed
                records_processed = 0
                stopped = False
                
                for future in concurrent.futures.as_completed(futures):  # Here's where you were passing a single future
                    if os.path.exists(stop_file):
                        stopped = True
                        safe_print(f"Stop file '{stop_file}' detected. Stopping the evaluation.", force=True)
                        break
                    
                    if verbose:
                        safe_print(f"evaluate_documents: into for future in concurrent.futures.as_completed(futures):")
                    
                    try:
                        result = future.result()
                        if DEBUG_MODE:
                            safe_print(f"Batch result type: {type(result)}, content: {result}")
                        
                        if isinstance(result, tuple):
                            if DEBUG_MODE:
                                safe_print(f"len(result[2]) {len(result[2])}")
                            inserted_recs += result[0]
                            updated_recs += result[1]
                            results.extend(result[2])  # Add results to the list
                            
                            records_processed += len(result[2])
                            
                            if records_processed % batch_size >= 0 or future == futures[-1]:  # Save every batch
                                # safe_print(f"{records_processed} % {batch_size}",records_processed % batch_size)
                                batch_number += 1
                                save_results(results, conn, verbose, number_of_batches, batch_number, batch_size, len(result[2]), workers, staging)
                                results = []  # Reset results after saving
                            else:
                                safe_print(f"{records_processed} % {batch_size}",records_processed % batch_size)
                                
                    except KeyboardInterrupt:
                        safe_print("Interrupted by user CTRL*C. Cleaning up...", force=True)
                        try:
                            executor.shutdown(wait=False)
                        except NameError:
                            pass  # Executor might not be available if KeyboardInterrupt was caught earlier

                        try:
                            # Close database connection if it's still open
                            conn.close()
                        except Exception:
                            pass

                        try:
                            # Assuming you're using Selenium or a similar tool
                            driver.quit()  # This should close the browser window
                        except NameError:
                            pass  # If 'driver' is not defined, just pass

                        cleanup()
                        return
                        
                    except Exception as e:
                        safe_print(f"An error occurred in a future: {e}", force=True)
                        # this will continue

                # Ensure any remaining results are saved if not a multiple of 100
                if results:
                    batch_number += 1
                    save_results(results, conn, verbose, number_of_batches, batch_number, batch_size, len(results), workers, staging)

                if stopped:
                    safe_print("Script stopped due to stop file. Cleaning up...", force=True)
                    for future in futures:
                        future.cancel()

                    try:
                        # Assuming you're using Selenium or a similar tool
                        driver.quit()  # This should close the browser window
                    except NameError:  # If 'driver' is not defined, just pass
                        pass

                    cleanup()
                    return                
                        
        if stopped:
                print(f"Script stopped 2 {stopped}")
            
    except KeyboardInterrupt:
        safe_print("Interrupted by user CTRL*C. Cleaning up...", force=True)
        cleanup()
        # sys.exit(0)  # Ensure the program exits
        return

    safe_print(f"\nFinal results: total tags for update {updated_recs} and insert (new) {inserted_recs}", force=True)
        
    return 
    

def evaluate_documents_new(conn, force, db_fields, query, batch_size, workers, staging, wait_time=10, verbose=False):
    global unique_number, doc_href_total, doc_href_inserted, big_total, start_batch_time, tag_cache
    
    cursor = conn.cursor()
    
    if verbose:
        safe_print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Evaluate_documents uses query {query}")
    
    if tag_cache is None:
        tag_cache = {}
        
    retry_execute(cursor, query)
    unique_docs = cursor.fetchall()
    safe_print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Unique documents registered in documents and not yet in documents_tags (based on URL): {len(unique_docs)}", force=True)
    
    results = return_all_registered_documents(conn)
    all_documents = results[0]
    documents_set = results[1]

    unique_number = 0
    doc_href_total = 0
    doc_href_inserted = 0
    inserted_recs = 0
    updated_recs = 0
    sql_return = []
    batch_sql_queries = []
    batch_size = int(batch_size)
    big_total = 0
    batch_number = 0
    number_of_workers = int(workers)
    
    wait_time = int(wait_time)
    wait_time_seconds = wait_time / 1000  # Convert milliseconds to seconds
    
    script_full_path = sys.argv[0]
    script_name, _ = os.path.splitext(script_full_path)
    stop_file = f"{script_name}.stop"
    
    if os.path.exists(stop_file):
        stopped = True
        safe_print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Stop file '{stop_file}' detected. Stopping the evaluation.", force=True)
        return

    # how many unique_docs (urls) to evaluate
    total_unique_docs = len(unique_docs)
    # how many batches in unique_docs
    number_of_batches = calculate_batches(total_unique_docs, batch_size)
    last_batch = total_unique_docs - ((number_of_batches-1) * batch_size)
    
    stopped = False
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=number_of_workers) as executor:  
            with requests.Session() as session:
                safe_print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting ThreadPoolExecutor with {number_of_workers} workers")
                
                batches = [unique_docs[i:i + batch_size] for i in range(0, len(unique_docs), batch_size)]
                futures = []
                batch_totals = 0
                
                # Dictionary to keep track of active workers
                active_workers = {}
                
                for i, batch in enumerate(batches):
                    safe_print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing batch {i + 1}")
                    # Only consider a worker as 'active' if there's room in the executor
                    if len(active_workers) < number_of_workers:
                        # If there's room, start the worker immediately
                        worker_id = f"Worker-{i+1}"
                        active_workers[worker_id] = True
                        safe_print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]-> Launching {worker_id} with {len(batch)} records. Active workers: {len(active_workers)}", force=True)
                        batch_totals += len(batch)
                        
                        future = executor.submit(process_doc_batch, batch, session, conn, staging, verbose, db_fields, all_documents, documents_set)
                        future.add_done_callback(lambda f, wid=worker_id: worker_done(f, wid, active_workers, verbose))
                        futures.append(future)
                        safe_print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Submitted task for {worker_id}")
                    else:
                        # If we've reached the max workers, wait for one to finish before starting another
                        start_wait = datetime.now()
                        while len(active_workers) >= number_of_workers:
                            time.sleep(0.1)  # Sleep for a short while to avoid busy-waiting
                            if (datetime.now() - start_wait).seconds > 10:  # If waiting for more than 10 seconds, log it
                                safe_print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Waiting for a worker slot. Current active workers: {len(active_workers)}")
                        # Now there's room, launch the worker
                        worker_id = f"Worker-{i+1}"
                        active_workers[worker_id] = True
                        safe_print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]-> Launching {worker_id} with {len(batch)} records. Active workers: {len(active_workers)}", force=True)
                        batch_totals += len(batch)
                        
                        future = executor.submit(process_doc_batch, batch, session, conn, staging, verbose, db_fields, all_documents, documents_set)
                        future.add_done_callback(lambda f, wid=worker_id: worker_done(f, wid, active_workers, verbose))
                        futures.append(future)
                        safe_print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Submitted task for {worker_id} after waiting")
                    
                    # Add delay between submissions if not the first batch
                    if i > 0:
                        safe_print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Waiting for {wait_time_seconds} seconds before next submission")
                        time.sleep(wait_time_seconds)

                results = []
                records_processed = 0
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        safe_print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Waiting for future result")
                        result = future.result()  # This will block until the future is complete
                        safe_print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Result received from a future")
                        if isinstance(result, tuple) and len(result) == 3:
                            inserted_recs += result[0]
                            updated_recs += result[1]
                            # Save immediately after a worker finishes
                            if len(result[2]) > 0:
                                batch_number += 1  # Increment batch number for each completed batch
                                save_results(result[2], conn, verbose, number_of_batches, batch_number, batch_size, last_batch, len(result[2]), workers, staging)
                                records_processed += len(result[2])
                                safe_print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]-> Batch {batch_number} processed {len(result[2])} records. Total processed: {records_processed}", force=True)
                        else:
                            safe_print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] evaluate_documents: unexpected future.result() length, {len(result)}")
                        
                        stopped = False
                    
                        if os.path.exists(stop_file):
                            stopped = True
                            safe_print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Stop file '{stop_file}' detected. Stopping the evaluation.", force=True)
                            break

                    except KeyboardInterrupt:
                        safe_print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Interrupted by user CTRL*C. Cleaning up...", force=True)
                        executor.shutdown(wait=False)
                        conn.close()
                        cleanup()
                        return

                    except Exception as e:
                        safe_print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] An error occurred processing results from a worker: {e}", force=True)

                safe_print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] All futures completed")

                if stopped:
                    safe_print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Script stopped due to stop file. Cleaning up...", force=True)
                    for future in futures:
                        future.cancel()
                    cleanup()
                    return                
                        
        if stopped:
            safe_print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Script stopped 2 {stopped}")
            
    except KeyboardInterrupt:
        safe_print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Interrupted by user CTRL*C. Cleaning up...", force=True)
        cleanup()
        return

    safe_print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Final results: total tags for update {updated_recs} and insert (new) {inserted_recs}", force=True)
        
    return


def worker_done(future, worker_id, active_workers, verbose):
    try:
        if verbose:
            safe_print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]-> {worker_id} completed. Active workers: {len(active_workers) - 1}", force=True)
        del active_workers[worker_id]
    except KeyError:
        if verbose:
            safe_print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]-> Attempted to remove {worker_id} from active workers but it was not present.", force=True)
       
def is_empty_list_array(input_list):
    """
    Check if the input is a list of lists where all elements are either empty lists or contain only empty strings.

    Args:
    input_list (list): The list to check.

    Returns:
    bool: True if the input is a list where all elements are lists containing only empty lists or empty strings, False otherwise.
    """
    # Check if input is a list
    if not isinstance(input_list, list):
        return False
    
    # Check if all elements are lists
    if not all(isinstance(item, list) for item in input_list):
        return False
    
    # Check if all nested lists contain only empty lists or empty strings
    for sublist in input_list:
        for item in sublist:
            if not isinstance(item, list) and item != '':
                return False
            if isinstance(item, list) and not is_empty_list_array(item):
                return False

    return True


def save_results(results, conn, verbose, number_of_batches, batch_number, batch_size, last_result, the_worker, staging):
    global big_total, start_batch_time, start_time
    batch_sql_queries = []
    seen_ids = set()

    def extract_id_and_query(sql):
        id_match = re.search(r"VALUES\s*\(\s*'([^']+)'", sql)
        if not id_match:
            id_match = re.search(r"WHERE\s+Id\s*=\s*'([^']+)'", sql)
        return id_match.group(1) if id_match else None, sql.strip()

    try:
        if not is_empty_list_array(results):
            if isinstance(results, list):
                for result in results:
                    if isinstance(result, (list, str)):
                        # Normalize all results to a list of SQL strings
                        sql_list = result if isinstance(result, list) else [result]
                        # when staging only add to batch
                        if staging:
                            batch_sql_queries.append(sql_list)
                        else:
                            for sql in sql_list:
                                if isinstance(sql, str):
                                    if staging:
                                        batch_sql_queries.append(query)
                                    else:
                                        id_value, query = extract_id_and_query(sql)
                                        if DEBUG_MODE:
                                            safe_print(f"id_value, query", id_value, query)
                                        if id_value and id_value not in seen_ids:
                                            seen_ids.add(id_value)
                                            batch_sql_queries.append(query)
                                        elif id_value:
                                            if DEBUG_MODE:
                                                safe_print(f"query_value double {query}", force=True)
                    elif verbose:
                        safe_print(f"save_results: Unexpected result type from {the_worker}: {type(result)}")
                        return

            if staging and len(batch_sql_queries) > 0:
                # write results to export file_name.staging.sql
                file_name = sys.argv[0].split('.')[0]
                # Write the error message to the file
                with open(f"STAGE_{datetime.now().strftime('%Y%m%d')}.staging.sql", "a") as staging_file:
                    safe_print(f"save_results: writing {len(batch_sql_queries)} results to file STAGE_{datetime.now().strftime('%Y%m%d')}.staging.sql")
                    for item in batch_sql_queries:
                        staging_file.write(f"{item[0]}\n")
                    
                staging_file.close()
                
            elif not staging and len(batch_sql_queries) > 0:
                # Execute SQL statements
                if DEBUG_MODE:
                    safe_print(f"save_results: unique save_results {len(batch_sql_queries)} batch_sql_queries")
                cursor = conn.cursor()
                try:
                    conn.execute("BEGIN")
                    for query in batch_sql_queries:
                        cursor.execute(query)
                    conn.commit()
                except sqlite3.Error as e:
                    conn.rollback()
                    safe_print(f"save_results: An error occurred: {e}")
                finally:
                    cursor.close()

            # Calculate and print timing information
            if len(batch_sql_queries) > 0:
                try:
                    end_time = datetime.now()
                    passed_time = end_time - start_batch_time
                    passed_full_time = end_time - start_time
                    big_total += len(batch_sql_queries)
                    batches_left = number_of_batches - batch_number
                    percentage_done = (batch_number/number_of_batches)*100
                    time_left = (passed_time.total_seconds() / batch_number) * batches_left

                    # Format time strings
                    def format_time(seconds):
                        days, remainder = divmod(int(seconds), 86400)
                        hours, remainder = divmod(remainder, 3600)
                        minutes, seconds = divmod(remainder, 60)
                        if days > 0:
                            return f"{days} days, {hours} hours, {minutes} minutes, {seconds} seconds"
                        elif hours > 0:
                            return f"{hours} hours, {minutes} minutes, {seconds} seconds"
                        elif minutes > 0:
                            return f"{minutes} minutes, {seconds} seconds"
                        else:
                            return f"{seconds} seconds"

                    result_text = format_time(passed_time.total_seconds())
                    lresult_text = format_time(time_left)

                    if time_left > 60:
                        safe_print(f"\n***  - unique queries in batch {the_worker}: {len(batch_sql_queries)}\n"
                                   f"***  - per second: {1/(passed_time.total_seconds() / big_total):.0f} records\n"
                                   f"***  - batches left: {batches_left} ({int(percentage_done)}% done)\n"
                                   f"***  - time passed: {result_text}\n"
                                   f"***  - records finished: {big_total}\n"
                                   f"***  - time left: {lresult_text}", force=True)
                    else:
                        safe_print(f"\n***  - unique queries in batch: {len(batch_sql_queries)}\n"
                                   f"***  - per second: {1/(passed_time.total_seconds() / big_total):.0f} records\n"
                                   f"***  - batches left: {batches_left} ({int(percentage_done)}% done)\n"
                                   f"***  - time passed: {result_text}\n"
                                   f"***  - records finished: {big_total}\n"
                                   f"***  - time left: {lresult_text}", force=True)
                except Exception as e:
                    safe_print(f"\n***  - unique queries in batch: {len(batch_sql_queries)} {e}")
            else:
                safe_print(f"\n\t\t- {len(results)}, type {type(results)}, resulted in no queries in batch_sql_queries\n{results}")

    except Exception as e:
        safe_print(f'save_results: something went wrong during preparing sql statements {e}')

    return

# #### #
def process_documents(href_list, documents_set, current_page_url):
    """
    Process a list of document links, checking if they already exist in the documents set.
    
    :param href_list: List of document lists to process
    :param documents_set: Set of URLs already present in the documents table
    :param current_page_url: URL of the current page for reference
    :return: List of new documents to be added to the database
    """
    new_documents = []
    existing_documents = []
    first_time = True
    
    for href in href_list:
        if DEBUG_MODE:
            safe_print('type(href)',type(href))
        if isinstance(href, list) and len(href) > 1:  # Ensure href is the list structure we expect
            
            if len(href) == 5:
                document_number, doc_type, doc_name, doc_date, doc_url = href
                if doc_url not in documents_set:
                    try:
                        document_id = doc_url.split('/')[-1]
                        document_name = unquote(document_id).split('.')[0]
                        
                        guid = str(uuid.uuid4())
                        
                        new_documents.append({
                            'Id': guid,
                            'Document ID': document_id,
                            'Document naam': doc_name,
                            'Locatie open.minvws.nl': doc_url, 
                            'FromFile': current_page_url
                            # 'Beoordeling': decision,
                            # 'DocumentPublicationDate': doc_date  # Use this if you want to store the date
                        })
                        documents_set.add(doc_url)  # Add to set to prevent duplicates in future iterations
                    except Exception as e:
                        if VERBOSE_MODE:
                            safe_print(f"Error processing href {href}: {e}")
                else:
                    # create list to update ModifyDate
                    if VERBOSE_MODE:
                        safe_print(f"Document {doc_url} is already registered.")

                    existing_documents.append({'Locatie open.minvws.nl': doc_url, 'FromFile': current_page_url, 'Bestandsnaam': doc_name, 'DocumentPublicationDate': doc_date})
                    
            # elif href.startswith('http'):
                
            else:
                header, documents = href[0], href[1]  # Split header and document data
                for document in documents:
                    if len(document) >= 5:  # Check if document list has enough elements
                        document_number, doc_type, doc_name, doc_date, doc_url = document
                        if doc_url not in documents_set:
                            try:
                                document_id = doc_url.split('/')[-1]
                                document_name = unquote(document_id).split('.')[0]
                                
                                guid = str(uuid.uuid4())
                                
                                new_documents.append({
                                    'Id': guid,
                                    'Document ID': document_id,
                                    'Document naam': doc_name,
                                    'Locatie open.minvws.nl': doc_url, 
                                    'FromFile': current_page_url
                                    # 'Beoordeling': decision,
                                    # 'DocumentPublicationDate': doc_date  # Use this if you want to store the date
                                })
                                documents_set.add(doc_url)  # Add to set to prevent duplicates in future iterations
                            except Exception as e:
                                if VERBOSE_MODE:
                                    safe_print(f"Error processing document {document}: {e}")
                        else:
                            if VERBOSE_MODE:
                                safe_print(f"Document {doc_url} is already registered.")
                            
                            existing_documents.append({'Locatie open.minvws.nl': doc_url, 'FromFile': current_page_url, 'Bestandsnaam': doc_name, 'DocumentPublicationDate': doc_date})
                            
                    else:
                        pass
                        # if VERBOSE_MODE:
                            # safe_print(f"Document list structure incorrect: {document}")
        elif isinstance(href, str):
            if first_time:
                grouped_href_list = [href_list[i:i+5] for i in range(0, len(href_list), 5)]
                for i, group in enumerate(grouped_href_list):
                    if len(group) != 5:
                        print(f"Warning: Group {i+1} does not have 5 elements: {group}")
                    else:
                        for group in grouped_href_list:
                            document_number, doc_type, doc_name, doc_date, doc_url = group
                            if doc_url not in documents_set:
                                try:
                                    document_id = doc_url.split('/')[-1]
                                    
                                    guid = str(uuid.uuid4())
                                    
                                    new_documents.append({
                                        'Id': guid,
                                        'Document ID': document_id,
                                        'Document naam': doc_name,
                                        'Locatie open.minvws.nl': doc_url, 
                                        'FromFile': current_page_url
                                        # 'Beoordeling': decision,
                                        # 'DocumentPublicationDate': doc_date  # Use this if you want to store the date
                                    })
                                    documents_set.add(doc_url)  # Add to set to prevent duplicates in future iterations
                                except Exception as e:
                                    if VERBOSE_MODE:
                                        safe_print(f"Error processing document href {doc_url}: {e}")
                            else:
                                if VERBOSE_MODE:
                                    safe_print(f"Document doc_url {doc_url} is already registered.")
                                existing_documents.append({'Locatie open.minvws.nl': doc_url})
                        first_time = False
                        return new_documents, existing_documents
            else:
                safe_print('Skipping', skipped)
    return new_documents, existing_documents

def parse_table_doc_refs(table):
    rows = table.find('tbody').find_all('tr')
    data = []
    if rows:
        for row in rows:
            cols = row.find_all('td')
            if len(cols) == 4:
                # print('number of cols', len(cols))
                # Extract text content
                row_data = [col.text.strip() for col in cols]
                # print('row_data', row_data)
                
                # Check for references (assuming references are always in an <a> tag within td)
                reference = None
                for col in cols:
                    # print(col)
                    link = col.find('a', class_='woo-a break-all')
                    if link:
                        reference = urljoin(base_href, link.get('href')) if not link.get('href').startswith('http') else link
                        break
                
                # Append reference if found
                
                if reference:
                    # print(reference)
                    row_data.append(reference)
                else:
                    row_data.append(' ')  # Append an empty string if no reference found
                
                # if DEBUG_MODE:
                    # print(f'\t\trow_data', row_data)
            
                data.append(row_data)
            else:
                if DEBUG_MODE:
                    safe_print("parse_table_doc_refs: len(cols) <> 4")
                    safe_print(row)
                    safe_print("------------------")
                    
                    safe_print(cols)
                
    return data

# returns tag_data for documents-section list
def getwooresponsivetable(tab_div, documents_div, current_url, pre_text, current_date, current_date_time, page_count=0, docs_found=0):
    
    table = tab_div.find('table', class_='woo-table')
    
    start_time_batch = datetime.now()

    if DEBUG_MODE:
        safe_print(f'\ngetwooresponsivetable: pre_text={pre_text}, page_count={page_count}, docs_found={docs_found}, current_url {current_url}')
    
    # check number of columns to avoid this step
    document_references = parse_table_doc_refs(table)
    tag_data = []
    
    if document_references:
        if docs_found and docs_found > 0:
            docs_start = docs_found + 1
        else:
            docs_start = 1
        
        number = docs_start
        record_number = 0
        total_documents = len(document_references)
        docs_found += total_documents
        
        if DEBUG_MODE:
            safe_print(f'getwooresponsivetable: document_references from parse_table_doc_refs: {total_documents}, docs_start: {docs_start}, docs_found: {docs_found}')
        
        try:
            for i in range(docs_start, docs_start + total_documents):  # Correct range to avoid out of range
                if i - 1 < (len(document_references) + docs_start):  # Check if index is within bounds
                    
                    record = document_references[record_number]
                    
                    # guid = str(uuid.uuid4())
                    
                    if isinstance(record, dict):
                        if all(key in record for key in ['Documentnummer', 'Type', 'Documentnaam', 'Datum']):
                            # Add link href data
                            tag_data.append({
                                'Id': str(uuid.uuid4()),
                                'Title': current_url,
                                'tag_key': f'{pre_text}linkhref_{number}',  # Use i instead of incrementing separately
                                'tag_value': record["Datum"],
                                'validSince': current_date,
                                'validUntil': '9999-12-31',
                                'CreatedDate': current_date_time
                            })
                            
                            # Add link name data
                            tag_data.append({
                                'Id': str(uuid.uuid4()),
                                'Title': current_url,
                                'tag_key': f'{pre_text}linkname_{number}',
                                'tag_value': record["Documentnaam"],
                                'validSince': current_date,
                                'validUntil': '9999-12-31',
                                'CreatedDate': current_date_time
                            })
                            
                            # Add link type
                            tag_data.append({
                                'Id': str(uuid.uuid4()),
                                'Title': current_url,
                                'tag_key': f'{pre_text}linktype_{number}',
                                'tag_value': record["Type"],
                                'validSince': current_date,
                                'validUntil': '9999-12-31',
                                'CreatedDate': current_date_time
                            })
                            
                            # Add link date
                            tag_data.append({
                                'Id': str(uuid.uuid4()),
                                'Title': current_url,
                                'tag_key': f'{pre_text}linkdate_{number}',
                                'tag_value': record["Datum"],
                                'validSince': current_date,
                                'validUntil': '9999-12-31',
                                'CreatedDate': current_date_time
                            })
                            
                            # Add link id data
                            tag_data.append({
                                'Id': str(uuid.uuid4()),
                                'Title': current_url,
                                'tag_key': f'{pre_text}id_{number}',
                                'tag_value': record["Documentnummer"],  # Use 'Documentnummer' key
                                'validSince': current_date,
                                'validUntil': '9999-12-31',
                                'CreatedDate': current_date_time
                            })
                            
                    else:
                        # safe_print(f"\t- Skipping record due to missing keys: {record}")
                        if isinstance(record, list):  # If record is a list, use indices
                            if len(record) > 4:
                                # Add link href data
                                tag_data.append({
                                    'Id': str(uuid.uuid4()),
                                    'Title': current_url,
                                    'tag_key': f'{pre_text}linkhref_{number}',
                                    'tag_value': record[4],
                                    'validSince': current_date,
                                    'validUntil': '9999-12-31',
                                    'CreatedDate': current_date_time
                                })
                                
                                # Add link name data
                                tag_data.append({
                                    'Id': str(uuid.uuid4()),
                                    'Title': current_url,
                                    'tag_key': f'{pre_text}linkname_{number}',
                                    'tag_value': record[2],
                                    'validSince': current_date,
                                    'validUntil': '9999-12-31',
                                    'CreatedDate': current_date_time
                                })
                                
                                # Add link type
                                tag_data.append({
                                    'Id': str(uuid.uuid4()),
                                    'Title': current_url,
                                    'tag_key': f'{pre_text}linktype_{number}',
                                    'tag_value': record[1],
                                    'validSince': current_date,
                                    'validUntil': '9999-12-31',
                                    'CreatedDate': current_date_time
                                })
                                
                                # Add link date
                                tag_data.append({
                                    'Id': str(uuid.uuid4()),
                                    'Title': current_url,
                                    'tag_key': f'{pre_text}linkdate_{number}',
                                    'tag_value': record[3],
                                    'validSince': current_date,
                                    'validUntil': '9999-12-31',
                                    'CreatedDate': current_date_time
                                })
                                
                                # Add link id data
                                tag_data.append({
                                    'Id': str(uuid.uuid4()),
                                    'Title': current_url,
                                    'tag_key': f'{pre_text}id_{number}',
                                    'tag_value': record[0],
                                    'validSince': current_date,
                                    'validUntil': '9999-12-31',
                                    'CreatedDate': current_date_time
                                })
                                
                            else:
                                safe_print(f"Skipping record due to insufficient data: {record}")
                        else:
                            safe_print(f"getwooresponsivetable: unexpected record type: {type(record)}, record: {record}")
                    
                    number += 1
                    record_number += 1
                else:
                    if DEBUG_MODE:
                        safe_print(f"! i - 1 < len(document_references): {i} - 1 < {len(document_references)}")
                    
        except Exception as e:
            safe_print(f"getwooresponsivetable: with docs, exception occured, {i}, len(document_references) {e}")
    else:
        total_documents = 0
        # not a woo-table with documents but with tags
        if table:
            # if verbose:
            try:
                rows = table.find_all('tr')
                if rows and len(rows) > 0:
                    for row in rows:
                        th = row.find('th')
                        td = row.find('td')
                        if th and td:
                            th_text = th.get_text(strip=True)
                            td_text = td.get_text(strip=False).replace('\n', ' ').replace(' +', ' ')
                            td_text = re.sub(r'\s+', ' ', td_text).strip()
                            tag_data.append({'Title': current_url, 'tag_key': f"{pre_text}{th_text}", 'tag_value': td_text})
                            
                            matches = re.findall(r'(\d{1,3}(?:,\d{3})*(?:\.\d{3})*)\s+(?:document(?:en)?|pagina(?:\'s))', td_text)
                            # Extract the numbers
                            if len(matches) == 2:
                                documents, pages = matches
                                tag_data.append({'Title': current_url, 'tag_key': f"{pre_text}{th_text}_documenten", 'tag_value': documents})
                                tag_data.append({'Title': current_url, 'tag_key': f"{pre_text}{th_text}_paginas", 'tag_value': pages})
                            elif len(matches) == 1:
                                paginas = matches[0]
                                tag_data.append({'Title': current_url, 'tag_key': f"{pre_text}{th_text}_paginas", 'tag_value': paginas})
                                
                            
                            links = td.find_all('a', href=True)
                            for i, link in enumerate(links, 1):
                                href = link.get('href')
                                if href:
                                    full_url = href if href.startswith(('http://', 'https://')) else urljoin(base_href, href)
                                    tag_data.append({'Title': current_url, 'tag_key': f"{pre_text}{th_text}_linkhref_{i}", 'tag_value': full_url})
                                    value = link.get_text(strip=True).replace('\n', ' ').replace(' +', ' ')
                                    value = re.sub(r'\s+', ' ', value).strip()
                                    tag_data.append({'Title': current_url, 'tag_key': f"{pre_text}{th_text}_linkname_{i}", 'tag_value': f"{value}"})
                                    
                else:
                    safe_print(f"{tabprefix}\t- cannot find any rows in table for {pre_text} and document {current_url}?")
                    
            except Exception as e:
                safe_print(f"getwooresponsivetable: no docs, error while adding tags for {pre_text}: {e}")
    
    if DEBUG_MODE:
        safe_print('page_count =', page_count, 'len(tag_data) =', len(tag_data), 'len(document_references) =', len(document_references))
    # safe_print(document_references)
    # exit(1)
    
    formatted_time = start_time_batch.strftime(f'%Y%m%d%H%M%S_{docs_found}')
    if WRITE_DEBUG_RESULTS:
        with open(f'{formatted_time}_document_references.txt', 'w', encoding='utf-8') as file:
            file.write(str(document_references))
                
    return tag_data, document_references, docs_found
    

def extract_and_process_links(html_content, url, session, table_name, documents_table, documents_set=None, conn=None, nodoc=False, in_batch=False, staging=False, pagination=True):
    global driver
    global doc_href_total
    global doc_href_inserted
    global doc_href_updated
    
    tags_found = 0
    tags_updated = 0
    tags_to_update = []
    docs_to_update = []
    existing_docs_to_update = []
    
    all_documents = [['Documentnummer', 'Type', 'Documentnaam', 'Datum', 'URL']]  
    all_tags = []
    sql_results = []
    tab_prefix = f"\t\t"
    docs_total = 0
    
    if not nodoc:
        try:
            if html_content is None:
                # safe_print(f"extract_and_process_links: html_content is None: fetching url")
                response = fetch_page(url, session)
                if response[0] == 200:
                    html_content = response[1]
                    if DEBUG_MODE:
                        safe_print(f"\nextract_and_process_links: received html_content two steps")
                else:
                    # safe_print(f"extract_and_process_links: Failed to get content from URL {url}")
                    return 0, 0, []
            else:
                if DEBUG_MODE:
                    safe_print(f"\nextract_and_process_links: received html_content one step")
            
            try:
                # safe_print(f"extract_and_process_links: type html_content {type(html_content)}")
                soup = html_content
            except Exception as e:
                # safe_print(f"extract_and_process_links: error parsing html_content")
                return 0, 0, []
                
            # safe_print(f"extract_and_process_links: got soup")

            def next_page_url(current_url, tab_id):
                try:
                    parsed = urlparse(current_url)
                    query_params = parse_qs(parsed.query)
                    if not tab_id == '<single_page>':
                        page_param = 'pu' if tab_id == 'tab1' else ('pa' if tab_id == 'tab2' else ('pn' if tab_id == 'tab3' else ( 'pu' if tab_id == 'single_page' else 'po')))
                        query_params[page_param] = [str(int(query_params.get(page_param, ['1'])[0]))]
                        return urlunparse(parsed._replace(query=urlencode(query_params, doseq=True)))
                    else:
                        return current_url
                    
                except Exception as e:
                    safe_print(f"next_page_url: exception {e}")
                    return ""

            def process_tab_content(tab_id, current_url, soup, page_count, docs_found):
                try:
                    # documents_div = soup.find('div', attrs={'data-e2e-name': 'documents-section'})
                    # if not documents_div:
                        # return [], []
                    
                    tab_div = soup.find('div', id=tab_id)
                    if not tab_div:
                        safe_print('div tab_id not found', tab_id)
                        return [], [], 0

                    current_date = datetime.now().strftime('%Y-%m-%d')
                    current_date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    if tab_id == 'tab1':
                        pre_text = 'ContainsDocument_dog_'  # deels openbaar gemaakt
                    elif tab_id == 'tab2':
                        pre_text = 'ContainsDocument_rop_'  # reeds openbaar
                    elif tab_id == 'tab3':
                        pre_text = 'ContainsDocument_nop_'  # niet openbaar
                    elif tab_id == 'tab4':
                        pre_text = 'ContainsDocument_nno_'  # nog niet openbaar
                    else:
                        pre_text = 'ContainsDocument_na_'   # specificatie niet aanwezig
                        
                    # docs_found should increase as long as the same base url is used
                    # return tag_data, document_references, total_documents
                    if DEBUG_MODE:
                        safe_print(f"{tab_prefix}- process_tab_content: getwooresponsivetable({tab_id}, '', current_url, pre_text={pre_text}, {current_date}, {current_date_time}, page_count={page_count}, docs_found={docs_found})")
                    results_table = getwooresponsivetable(tab_div, "", current_url, f'{pre_text}', current_date, current_date_time, page_count, docs_found)
                    
                    if len(results_table) >= 2:
                        # tag_data document_references #total_documents 
                        if DEBUG_MODE:
                            safe_print(f"{tab_prefix}- process_tab_content: returning content: {len(results_table[0])}")
                        return results_table
                    else:
                        safe_print(f"{tab_prefix}- process_tab_content: returning content unexpected")
                        return [], [], 0
                        
                except Exception as e:
                    safe_print(f"process_tab_content: exception {e}")
                    
                return [], [], 0

            # Determine which tabs are present
            tab_ids = ['tab1', 'tab2', 'tab3', 'tab4']
            
            
            try:
                documents_section = soup.find('h2', id="documenten")
            except Exception as e:
                safe_print(f"extract_and_process_links: error soup.find('h2', id='documenten')")
                pass
                
            if documents_section:
                tab_list = soup.find('ul', class_='woo-tab-list')   # are there tab headers?
                if tab_list:
                    for tab_id in tab_ids:
                        tab_div = soup.find('div', id=tab_id)
                        if tab_div:
                            current_url = url  # Reset back to the starting URL for each tab
                            page_count = 1  # start at page one for every tab
                            total_tags = 0
                            nextPage = True
                            
                            while nextPage: # loop through this tab
                                if VERBOSE_MODE:
                                    safe_print(f"{tab_prefix}- extract_and_process_links: Processing {tab_id}, page {page_count}")
                                
                                try:
                                    # return tag_data, document_references, total_documents
                                    if DEBUG_MODE:
                                        safe_print(f"{tab_prefix}- process_tab_content(tab_id={tab_id}, {current_url}, soup, page_count={page_count}, docs_total={docs_total})")
                                    tags, documents, total_docs = process_tab_content(tab_id, url, soup, page_count, docs_total)
                                except Exception as e:
                                    safe_print(f"{tab_prefix}- extract_and_process_links: error in tags, documents = process_tab_content({tab_id}, {current_url}, soup, {page_count})")
                                    break
                                    
                                msg = ""
                                if tags:
                                    for tag in tags:
                                        total_tags += len(tag)
                                    all_tags.append(tags)
                                    msg += f"{tab_prefix}- tags 0: {total_tags}/{len(all_tags)}\n"
                                
                                if documents:
                                    all_documents.extend(documents)
                                    msg += f"{tab_prefix}- docs 0: {len(documents)}/{len(all_documents)}\n"
                                
                                if total_docs:
                                    docs_total = total_docs
                                    msg += f"{tab_prefix}- total_docs / docs_total 0: {total_docs}/{docs_total}"
                                
                                if DEBUG_MODE:
                                    safe_print(f"{tab_prefix}- in_batch: {in_batch}\n{msg}")

                                # Check for next page for this specific tab
                                tab_content = soup.find('div', id=tab_id, attrs={'data-tab-content': ''})
                                if tab_content:
                                    pagination_nav = tab_content.find('nav', attrs={'data-e2e-name': 'pagination'})
                                    if pagination_nav:
                                        next_link = pagination_nav.find('a', rel='next')
                                        if next_link:
                                            href = next_link.get('href')
                                            full_url = href if href.startswith(('http://', 'https://')) else urljoin(url, href)
                                            if full_url:
                                                current_url = next_page_url(full_url, tab_id)
                                                if DEBUG_MODE:
                                                    safe_print(f"{tab_prefix}- extract_and_process_links: got {docs_total} docs, next page url for {tab_id} {current_url}")
                                                page_count += 1
                                                # retrieve the next page
                                                response = fetch_page(current_url, session)
                                                if response[0] != 200:
                                                    safe_print(f"{tab_prefix}- extract_and_process_links: reading next page url for {tab_id} error {response[0]}")
                                                    nextPage = False
                                                    break
                                                
                                                soup = response[1] # BeautifulSoup(response[1], 'html.parser')
                                                document_section = soup.find('div', attrs={'data-e2e-name': 'documents-section'})
                                                if document_section:
                                                    table_div = document_section.find('div', class_='woo-responsive-table')
                                                    if table_div:
                                                        soup = document_section
                                                        # soup = document_section # table_div
                                                        pass
                                                    else:
                                                        safe_print(f"{tab_prefix}- extract_and_process_links: Could not find 'woo-responsive-table' within document section for {tab_id}")
                                                        nextPage = False
                                                else:
                                                    safe_print(f"{tab_prefix}- extract_and_process_links: Could not find document section for {tab_id}")
                                                    nextPage = False
                                                    
                                            else:
                                                safe_print(f"{tab_prefix}- extract_and_process_links: no next page url extracted for {tab_id}")
                                                nextPage = False
                                        else:
                                            nextPage = False
                                    else:
                                        if DEBUG_MODE:
                                            safe_print(f"{tab_prefix}- extract_and_process_links: no pagination found for {tab_id}")
                                        nextPage = False
                                else:
                                    if VERBOSE_MODE:
                                        safe_print(f"{tab_prefix}- extract_and_process_links: content for tab {tab_id} not found")
                                    nextPage = False
                                    current_url = url
                        else:
                            if VERBOSE_MODE:
                                safe_print(f"{tab_prefix}- Tab {tab_id} not found, skipping.")    
                else:
                    # If no tabs, process the document section directly
                    current_date = datetime.now().strftime('%Y-%m-%d')
                    current_date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    pre_text = 'ContainsDocument_na_'
                    if DEBUG_MODE:
                        safe_print(f"extract_and_process_links: no tab list, into while nextPage")
                    page_count = 1
                    current_url = url
                    nextPage = True
                    total_tags = 0
                    tab_id = "<single_page>"
                    
                    while True:
                        # return tag_data, document_references, total_documents
                        if DEBUG_MODE:
                            safe_print(f"{tab_prefix}- extract_and_process_links: Processing {tab_id}, page {page_count} {current_url}")
                        
                        tags, documents, total_docs = getwooresponsivetable(soup, "", current_url, f'{pre_text}', current_date, current_date_time, page_count, docs_total)
                        
                        msg = ""
                        if tags:
                            for tag in tags:
                                total_tags += len(tag)
                            all_tags.extend(tags)
                            msg += f"tags 1: {total_tags}/{len(all_tags)}\n"
                        if documents:
                            all_documents.extend(documents)
                            # this will nest results
                            msg += f"docs 1: {len(documents)}/{len(all_documents)}\n"
                        if total_docs:
                            docs_total = total_docs
                            msg += f"total_docs / docs_total 1: {total_docs}/{docs_total}\n"
                        if DEBUG_MODE:
                            safe_print(f"{msg}")
                        
                        if not nextPage:
                            break
                            
                        # navigate
                        pagination_nav = soup.find('nav', attrs={'data-e2e-name': 'pagination'})
                        if pagination_nav and nextPage:
                            next_link = pagination_nav.find('a', rel='next')
                            if next_link:
                                href = next_link.get('href')
                                full_url = href if href.startswith(('http://', 'https://')) else urljoin(url, href)
                                if full_url == current_url:
                                    break
                                    
                                if full_url:
                                    current_url = next_page_url(full_url, tab_id)
                                    page_count += 1
                                    if DEBUG_MODE:
                                        safe_print(f"{tab_prefix}- extract_and_process_links: got {docs_total} docs, next page {page_count} url for {tab_id} {current_url}")
                                    
                                    # retrieve the next page
                                    response = fetch_page(current_url, session)
                                    if response[0] != 200:
                                        safe_print(f"{tab_prefix}- extract_and_process_links: reading next page url for {tab_id} error {response[0]}")
                                        nextPage = False
                                        break
                                    else:
                                        try:
                                            # nextPage = True
                                            if DEBUG_MODE:
                                                safe_print(f"extract_and_process_links: fetch_page {page_count} result {response[0]} ")
                                            
                                            soup = response[1] # BeautifulSoup(response[1], 'html.parser')
                                            document_section = soup.find('div', attrs={'data-e2e-name': 'documents-section'})
                                            if document_section:
                                                table_div = document_section.find('div', class_='woo-responsive-table')
                                                if table_div:
                                                    # Here you have your table_div, proceed with your logic
                                                    soup = document_section # table_div
                                                    pass
                                                else:
                                                    safe_print(f"{tab_prefix}- extract_and_process_links: Could not find 'woo-responsive-table' within document section for {tab_id}")
                                                    nextPage = False
                                            else:
                                                safe_print(f"{tab_prefix}- extract_and_process_links: Could not find document section for {tab_id}")
                                                nextPage = False
                                                
                                        except Exception as e:
                                            safe_print(f"extract_and_process_links: nextPage issues, {e}")
                                            nextPage = False
                                    # soup = response[1] # BeautifulSoup(response[1], 'html.parser')
                                else:
                                    safe_print(f"{tab_prefix}- extract_and_process_links: no next page url extracted for {tab_id}")
                                    nextPage = False
                            else:
                                if DEBUG_MODE:
                                    safe_print(f"extract_and_process_links: pagination_nav not found {page_count}")
                                # nextPage = False
                                break
                        else:
                            page_count += 1
                            if DEBUG_MODE:
                                safe_print(f"{tab_prefix}- extract_and_process_links: no (next) pagination found on {page_count} for {tab_id}")
                                
                            nextPage = False
                        
            else:
                if VERBOSE_MODE:
                    safe_print(f"\n\t\t- no documents section found on {url}\n{type(soup)}")
                    exit(1)
            
            if DEBUG_MODE:
                safe_print(f"OUT OF LOOP")
                
            # Process tags and documents
            if all_tags and conn is not None:
                if VERBOSE_MODE:
                    safe_print(f"Processing records in all_tags with update_or_insert_tag")
                n = 0
                try:
                    for record in all_tags:
                        n += 1
                        # safe_print(n, type(record))
                        if record and isinstance(record, dict):  # Ensure record isn't None, Batch processing
                            results = update_or_insert_tag(conn, table_name, record, staging, VERBOSE_MODE, True)
                            if results:
                                tags_found += results[0]
                                tags_updated += results[1]
                                sql_results.extend(results[2])
                except Exception as e:
                    safe_print(f"Handling all_tags raised an exception, {e}")

            if not nodoc and not in_batch:
                safe_print(f"all_documents {len(all_documents)-1}")
                # href_list = [row for row in all_documents[1:] if row and len(row) > 0 and row[-1]]  # Ensure row exists and has a URL
                href_list = extract_urls(all_documents)
                
                if DEBUG_MODE:
                    safe_print(f"href_list {len(href_list)}")
                if href_list:
                    # returns the TAGS for the new documents in new_docs
                    # existing_docs holds the identifiers for the existing documents
                    new_docs, existing_docs = process_documents(href_list, documents_set, url)
                    if new_docs:
                        docs_to_update.extend(new_docs)
                    if existing_docs:
                        existing_docs_to_update.extend(existing_docs)
                        
                    # Insert documents
                    cursor = conn.cursor()
                    for document in docs_to_update:
                        if document:  # Ensure document isn't None
                            insert_sql = """
                            INSERT INTO documents 
                            ([Id], [Locatie open.minvws.nl], [Document naam], FromFile, [Document ID], ModifiedDate, CreatedDate) 
                            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                            """
                            try:
                                cursor.execute(insert_sql, (
                                    document['Id'],
                                    document['Locatie open.minvws.nl'], 
                                    document['Document naam'], 
                                    url, 
                                    document['Document ID']
                                ))
                                doc_href_inserted += 1
                            except sqlite3.Error as e:
                                safe_print(f"\tDatabase error while inserting document: {e}")
                    
                    for document in existing_docs_to_update:
                        if document:  # Ensure document isn't None
                            # {'Locatie open.minvws.nl': doc_url, 
                            # 'FromFile': current_page_url, 
                            # 'Bestandsnaam': doc_name, 
                            # 'DocumentPublicationDate': doc_date}
                            update_sql = """UPDATE documents SET ModifiedDate = CURRENT_TIMESTAMP, FromFile = ?, Bestandsnaam = ?, DocumentPublicationDate = ? WHERE [Locatie open.minvws.nl] = ?"""
                            try:
                                cursor.execute(update_sql, (
                                    document['FromFile'],
                                    document['Bestandsnaam'],
                                    document['DocumentPublicationDate'],
                                    document['Locatie open.minvws.nl']
                                ))
                                doc_href_updated += 1
                            except sqlite3.Error as e:
                                safe_print(f"\tDatabase error while inserting document: {e}")
                                
                    conn.commit()
                    
            elif not nodoc and in_batch:
                # tags_updated += results[1]
                href_list = extract_urls(all_documents)
                # href_list = [row[-1] for row in all_documents[1:] if row and len(row) > 0 and row[-1]]  # Ensure row exists and has a URL
                if DEBUG_MODE:
                    safe_print(f"in_batch, waiting with evaluating all_documents (docs_total) {docs_total} (total_docs) {total_docs} with len(all_documents)-1: {len(all_documents)-1}, tags found/created {tags_found}, existing tags updated {tags_updated}, sql_results {len(sql_results)}")

            # if VERBOSE_MODE:
                # safe_print(f"Tags found: {tags_found}, Tags updated: {tags_updated}")
                # safe_print(f"Documents inserted: {len(docs_to_update)}")

        except Exception as e:
            safe_print(f"Error in extract_and_process_links {url}: {e}")
            pass

    # Ensure we don't try to get length of None
    # href_list = [row[-1] for row in all_documents[1:] if row and len(row) > 0 and row[-1]]  # Ensure row exists and has a URL
    href_list = extract_urls(all_documents)
    
    return len(href_list), all_documents, all_tags if all_tags is not None else [], sql_results if sql_results is not None else []

    
def return_all_registered_documents(conn):
    cursor = conn.cursor()
    try:
        cursor.execute("""SELECT DISTINCT [Locatie open.minvws.nl] FROM documents INDEXED BY [idx_documents_Locatie_open_minvws_nl] """)
    except sqlite3.Error as e:
        print(f"SQLite error selecting documents: {e}")
        conn.rollback()
        return {}, [] 
        
    all_documents = {doc[0]: doc for doc in cursor.fetchall()}
    
    documents_set = set(doc[0] for doc in all_documents.values())
    
    return all_documents, documents_set

def extract_urls(nested_list):
    """
    Extracts all URLs from a deeply nested list structure.
    
    :param nested_list: A list containing nested lists where URLs are stored
    :return: A list of all URLs found
    """
    urls = []
    
    # Check if the list is not empty and has the expected structure
    if nested_list and all(isinstance(item, list) for item in nested_list):
        for outer_item in nested_list:
            if outer_item and isinstance(outer_item[0], list):  # Check for header and documents
                header, documents = outer_item[0], outer_item[1:]  # Separate header from documents
                for document_group in documents:
                    if isinstance(document_group, list):
                        for document in document_group:
                            if isinstance(document, list) and len(document) >= 5:
                                # Assuming the URL is the last element in each document list
                                url = document[-1]
                                if isinstance(url, str):  # Ensure we're dealing with a string URL
                                    urls.append(url)
                            else:
                                # safe_print(document)
                                if isinstance(document, str):
                                    if document.startswith('http'):
                                        urls.append(document)
                                else:
                                    safe_print(f"type(document): {type(document)} - {document}")
                    else:
                        safe_print(f"type(document_group): {type(document_group)}")
            elif isinstance(outer_item, list):
                if outer_item == ['Documentnummer', 'Type', 'Documentnaam', 'Datum', 'URL']:
                    pass
                else:
                    url = outer_item[-1]
                    if isinstance(url, str):  # Ensure we're dealing with a string URL
                        urls.append(url)
                    # safe_print(f"outer_item: {outer_item}")
            else:
                safe_print(f"type(outer_item): {type(outer_item)}")
    else:
        safe_print(f"Not all items in nested_list are lists?")
        
    return urls       

def get_unique_titles(your_list):
    unique_titles = set()
    
    for sublist in your_list:
        for item in sublist:
            # Check if item is a dictionary before trying to get 'Title'
            if isinstance(item, dict):
                title = item.get('Title')
                if title:
                    unique_titles.add(title)
            elif isinstance(item, list):  # If item is a list, dive deeper
                for nested_item in item:
                    if isinstance(nested_item, dict):
                        title = nested_item.get('Title')
                        if title:
                            unique_titles.add(title)
    
    return list(unique_titles)
        
def handle_request_pages(conn, session, staging, query=None, nodoc=False, in_batch=False):
    global big_total
    global doc_href_updated
    cursor = conn.cursor()
    try:
        cursor.execute(query)
    except sqlite3.Error as e:
        print(f"SQLite error executing query: {e}")
        conn.rollback()
        return  # or handle this error in a way that makes sense for your application

    rows = cursor.fetchall()
    total = len(rows)
    i = 0
    
    grand_total_inserted = 0
    grand_total_updated = 0
    
    big_total = 0

    results = return_all_registered_documents(conn)
    all_documents = results[0]
    documents_set = results[1]
    
    safe_print(f"Records in query {total}, registered documents {len(all_documents)}")
    
    # Create Session
    
    doc_href_total = 0
    doc_href_inserted = 0
    doc_href_updated = 0
    
    total_docs = 0
    total_sql = 0
    total_tags = 0
    
    sql_return = []
    
    sql_results = []
    document_references = []
    request_tags = []
    detailed_tags = []
    
    title_tags = {}
    
    new_documents = []
    new_documents_sql = []
    update_documents_sql = []
    
    tab_prefix = f"\t"
    href_list = ""

    inserted_recs = 0
    updated_recs = 0
    
    start_time_batch = datetime.now()
    formatted_time = start_time_batch.strftime('%Y%m%d%H%M%S')
    
    for row in rows:
        
        try:
            title, dossier_url = row
            # print(f"Fetching {dossier_url} ")
        except ValueError:
            print(f"Error: Unexpected row structure for {row}")
            continue
            
        i += 1
        

        current_url = dossier_url
        page_count = 1
        last_one = False
        # result = fetch_page(current_url, session)
        safe_print(f"\n({i}/{total}) Fetching page {page_count} for {current_url} ...")
        start_time_request = datetime.now()
        
        # while True:
        try:
            result = fetch_page(current_url, session)
            if result:
                return_status, detailed_soup = result
                # Check for next page on request page
                next_page_link = detailed_soup.find('a', attrs={'rel': 'next'})  # Adjust selector as per actual HTML
                if next_page_link and 'href' in next_page_link.attrs:
                    safe_print(f"\n{tab_prefix}- page {page_count} ({return_status}) for {current_url} ...")
                    last_one = False
                else:
                    safe_print(f"\n{tab_prefix}- last page of {page_count} ({return_status}) for {current_url} ...")
                    last_one = True
                    
                # Process the page
                # detailed_tags = extract_all_tags_from_soup(detailed_soup, current_url, session, datetime.now().strftime('%Y-%m-%d'), VERBOSE_MODE, nodoc, all_documents, documents_set, i)
                # safe_print(f"\n({i}/{total})\nFetching ({return_status}) {current_url} ...")

                try:
                    # this one contains extract_and_process_links
                    # return tags, document_references, sql_results
                    detailed_tags = extract_all_tags_from_soup(detailed_soup, dossier_url, session, datetime.now().strftime('%Y-%m-%d'), VERBOSE_MODE, nodoc, all_documents, documents_set, i, page_count)
                    
                    if DEBUG_MODE and detailed_tags[0]:
                        safe_print(f"handle_request_pages: after extract_all_tags_from_soup, tags: {len(detailed_tags[0])}, {type(detailed_tags[0])}, and docs: {len(detailed_tags[1])}, {type(detailed_tags[1])}")
                        omvang_value = next((tag['tag_value'] for tag in detailed_tags[0] if tag['tag_key'] == 'Omvang openbaarmaking_documenten'), '0')
                        
                    elif not VERBOSE_MODE and detailed_tags[0]:
                        omvang_value = next((tag['tag_value'] for tag in detailed_tags[0] if tag['tag_key'] == 'Omvang openbaarmaking_documenten'), '0')
                        
                    # elif VERBOSE_MODE:
                        # for tag in detailed_tags:
                            # safe_print(f"{i} detailed_tags[..]: {len(tag)}")
                    elif detailed_tags[0]:
                        omvang_value = next((tag['tag_value'] for tag in detailed_tags[0] if tag['tag_key'] == 'Omvang openbaarmaking_documenten'), '0')
                        
                    else:    
                        omvang_value = ""
                            
                    if detailed_tags[1] is not None:
                        count_documents = 0
                        for doc in detailed_tags[1]:
                            count_documents += len(doc)
                        if nodoc:
                            safe_print(f"\t- extract all tags returned {len(detailed_tags[0])} tags and no documents because of --nodoc switch")
                        else:
                            safe_print(f"\t- extract all tags returned {len(detailed_tags[0])} tags and {count_documents-1} documents")
                    else:
                        safe_print(f"\t- extract all tags returned {len(detailed_tags[0])} tags and no documents")
                    
                    try:
                        end_time = datetime.now()
                        safe_print(f"\t- with request containing {omvang_value} documents")
                        safe_print(f"\t- finished in {end_time-start_time_request}")
                    except Exception as e:
                        safe_print(f"with request containing {omvang_value} documents")
                        pass

                    if detailed_tags[0]:
                        request_tags.append(detailed_tags[0])
                    if detailed_tags[1]:
                        # for doc in detailed_tags[1]:
                        document_references.extend(detailed_tags[1])
                    if detailed_tags[2]:
                        sql_results.extend(detailed_tags[2])
                        
                        # print(f"\t- extract all tags returned {len(detailed_tags)} tags")
                except Exception as e:
                    safe_print(f"Error extracting tags from soup for {title}, {dossier_url}: {e}", force=True)
                    script_name = sys.argv[0].split('.')[0]
                    # Write the error message to the file
                    with open(f"{script_name}.undetected", "a") as error_file:
                        error_file.write(f"{dossier_url}\n")
                        
                    continue  # Skip to the next row if tag extraction fails
            
                # Check for next page
                # next_page_link = detailed_soup.find('a', attrs={'rel': 'next'})  # Adjust selector as per actual HTML
                # if next_page_link and 'href' in next_page_link.attrs:
                    # current_url = urljoin(dossier_url, next_page_link['href'])
                    # page_count += 1
                    # if VERBOSE_MODE:
                        # safe_print(f"handle_request_pages: next page {page_count} with {current_url}")
                    
            else:
                safe_print(f"Failed to fetch page for {title}, {current_url}")
                # break
                
            # if last_one:
                # break
                
        except requests.RequestException as e:
            safe_print(f"Failed to fetch detailed page for {title}, {current_url}: {e}")
            # break

        if DEBUG_MODE:
            safe_print('document_references',document_references)
        for doc in document_references:
            total_docs += len(doc)
        for sql in sql_results:
            total_sql += len(sql)
        for tag in detailed_tags:
            total_tags += len(tag)
        
        if VERBOSE_MODE:
            safe_print(f"request_tags {len(request_tags)}")
            safe_print(f"document_references {total_docs} in {len(document_references)} embedded lists")
            # safe_print(f"document_references[0] {len(document_references[0])}")
            safe_print(f"sql_results {total_sql} in {len(sql_results)} embedded lists")
            safe_print(f"detailed_tags {total_tags} in {len(detailed_tags)} embedded lists")
        
    # print(detailed_tags + download_tags)
    # return
    
    end_time = datetime.now()
    safe_print(f"\n... end of discovery, elapsed time {end_time-start_time_batch}.")
    safe_print(f"\nContinuing with database operations ...")
    start_time_db = datetime.now()
    
    
    if request_tags:
        n = 0
        l = 0
        for tag in request_tags:
            n += 1
            l += len(tag)
            
        if WRITE_RESULTS:
            with open(f'{formatted_time}_request_tags.txt', 'w', encoding='utf-8') as file:
                file.write(str(request_tags))

        unique_titles = get_unique_titles(request_tags)

        safe_print(f"\t- {n} embedded lists resulted in {l} tags for {len(unique_titles)} unique requests")        
        
        with sqlite3.connect(args.db, isolation_level=None, check_same_thread=False) as cache_conn:
            for title in unique_titles:
                title_tags[title] = fetch_all_tags_for_title('requests_tags', title, VERBOSE_MODE)
        
        if VERBOSE_MODE:
            safe_print(f"\t\tExisting tags for {len(title_tags)} unique requests ")
            
        for tags in request_tags:
            try:
                # Debug: Print the tag being processed
                # safe_print('type(tags)',type(tags))
                for tag in tags:
                    # safe_print('type(tag)',type(tag))
                    if DEBUG_MODE:
                        safe_print(f"Processing tag: {type(tag)} {tag}")
                        
                    if isinstance(tag, list):
                        # for element in tag:
                            # # result = update_or_insert_tag(conn, 'requests_tags', tag, staging, VERBOSE_MODE, True)
                            # result = update_or_insert_tag(conn, 'requests_tags', element, staging, VERBOSE_MODE, in_batch, existing_tag_values=title_tags[dossier_url])
                            # if result:
                                # inserted_recs += result[0]
                                # updated_recs += result[1]
                                # if result[0] > 0 or result[1] > 0:
                                    # sql_return.append(result[2])
                            # if DEBUG_MODE:
                                # safe_print(f"Successfully processed element: {element['tag_key']} = {element['tag_value']}")
                        pass
                    elif isinstance(tag, dict):
                        dossier_url = tag['Title']
                        result = update_or_insert_tag(conn, 'requests_tags', tag, staging, VERBOSE_MODE, in_batch, existing_tag_values=title_tags[dossier_url])
                        if result:
                            inserted_recs += result[0]
                            updated_recs += result[1]
                            if result[0] > 0 or result[1] > 0:
                                sql_return.append(result[2])
                        if VERBOSE_MODE:
                            safe_print(f"Successfully processed tag: {tag['tag_key']} = {tag['tag_value']}")
                
            except sqlite3.Error as e:
                # Debug: More detailed error message for SQLite specific errors
                safe_print(f"SQLite error while processing tag {tag['tag_key']}: {e}")
                safe_print(f"Error details for {title}, {dossier_url}")
                conn.rollback()
            except Exception as e:
                # Debug: Detailed error message for unexpected errors
                safe_print(f"Unexpected error while processing tag {tag['tag_key']}: {e}")
                safe_print(f"Error details for {title}, {dossier_url}: {str(tag)}")
                
                # Additional debug: Check the structure of tag
                safe_print(f"Tag structure: keys = {list(tag.keys())}, values = {list(tag.values())}\n")
                
                conn.rollback()
    
    safe_print(f"\t- which resulted in {inserted_recs} inserted and {updated_recs} updated tags")
    
    if document_references and len(document_references) > 0:
        # evaluate documents
        # get href_list, get documents_set 
        results = return_all_registered_documents(conn)
        # all_documents = results[0]
        documents_set = results[1]
        # href_list consists of the newly scraped url's
        # href_list = [row[4] for row in document_references]
        # href_list = [row[-1] for row in document_references[1:] if row and len(row) > 0 and row[-1]]  

        doc_list = []
        
        for document in document_references:
            n = 0
            doc_href_total += len(document)
            for record in document:
                if n > 0:
                    doc_list.extend(record)
                else:
                    doc_href_total = doc_href_total - 1
                    n += 1
        
        if WRITE_RESULTS:
            with open(f'{formatted_time}_document_references_full.txt', 'w', encoding='utf-8') as file:
                file.write(str(document_references))

        # href_list = extract_urls(document_references)
        # href_list = document_references
        
        if DEBUG_MODE:
            safe_print("-----------------------------------------------")
            safe_print(doc_list)
        
        doc_href_total += len(document_references)
        
        href_list = [row[-1] for row in document_references]
        
        # Save the content to local cache
        if WRITE_RESULTS:
            with open(f'{formatted_time}_href_list_full.txt', 'w', encoding='utf-8') as file:
                file.write(str(href_list))
        
        # returns the TAGS for the new documents in new_docs
        # existing_docs holds the identifiers for the existing documents
        new_docs, existing_docs = process_documents(doc_list, documents_set, dossier_url)
        # if new_docs:
            # docs_to_update.extend(new_docs)
        # if existing_docs:
            # existing_docs_to_update.extend(existing_docs)
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if new_docs:
            # Insert documents
            # Save the content to local cache
            if WRITE_RESULTS:
                with open(f'{formatted_time}_new_docs.txt', 'w', encoding='utf-8') as file:
                    file.write(str(new_docs))
            new_documents.extend(new_docs)
            
            if not True:
                for document in new_docs:
                    if document:  # Ensure document isn't None
                        if VERBOSE_MODE:
                            if isinstance(document, dict):
                                safe_print(f"Document keys: {list(document.keys())[:5]}")  # Print first 5 keys
                            elif isinstance(document, str):
                                safe_print(f"Document content: {document[:100]}")
                            else:
                                safe_print(f"Document type: {type(document)}")

                        try:
                            if document.get('Locatie open.minvws.nl') != 'URL':  # Use .get() for safety
                                insert_sql = """
                                INSERT INTO documents 
                                (Id, [Locatie open.minvws.nl], [Document naam], FromFile, [Document ID], ModifiedDate, CreatedDate) 
                                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                                """
                                formatted_sql = insert_sql.replace("?", "'{}'").format(document['Id'], document['Locatie open.minvws.nl'], document['Document naam'], document['FromFile'], document['Document ID'])
                                new_documents_sql.append(formatted_sql)
                                doc_href_inserted += 1
                        except KeyError as ke:
                            safe_print(f"\tKeyError occurred: {ke}. Document might be missing required fields.")
                        except Exception as e:
                            safe_print(f"\tAn exception occurred gathering SQL statements: {e}")
            else:
                safe_print(f"\t- collected {len(new_docs)} new documents, total {len(new_documents)}")
                
            if WRITE_RESULTS:
                with open(f'{formatted_time}_new_docs.sql', 'w', encoding='utf-8') as file:
                    file.write(str(new_documents_sql))
        else:
            if VERBOSE_MODE:
                safe_print(f"process_documents returned no new_docs for {len(new_docs)} references")        
        
        if existing_docs:
            for document in existing_docs:
                if document:  # Ensure document isn't None
                    update_sql = """UPDATE documents SET ModifiedDate = CURRENT_TIMESTAMP WHERE [Locatie open.minvws.nl] = ?"""
                    try:
                        formatted_sql = update_sql.replace("?", "'{}'").format(document['Locatie open.minvws.nl'])
                        update_documents_sql.append(formatted_sql)
                        
                        doc_href_updated += 1
                    except sqlite3.Error as e:
                        safe_print(f"\tDatabase error while updating document: {e}")
        else:
            if VERBOSE_MODE:
                safe_print(f"process_documents returned no existing docs for {len(existing_docs)} references")    
            
        
        # exit(1)
    
    grand_total_inserted  += inserted_recs
    grand_total_updated += updated_recs
    
    if sql_return:
        if len(sql_return) > 0:
            # def save_results(results, conn, verbose, number_of_batches, batch_number, batch_size, last_result, the_worker, staging):
            if VERBOSE_MODE:
                safe_print(f"\nSaving {len(sql_return)} new tags {staging}")
            save_results(sql_return, conn, VERBOSE_MODE, 1, 1, len(sql_return), 1, 1, staging)
    
    if new_documents:
        safe_print(f"\t- inserting new documents using executemany")
        cursor = conn.cursor()
        insert_sql = """
        INSERT INTO documents 
        (Id, [Locatie open.minvws.nl], [Document naam], FromFile, [Document ID], ModifiedDate, CreatedDate) 
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """
        params_list = [(doc['Id'], doc['Locatie open.minvws.nl'], doc['Document naam'], doc['FromFile'], doc['Document ID']) for doc in new_docs]
        try:
            cursor.executemany(insert_sql, params_list)
            conn.commit()
        except sqlite3.Error as e:
            safe_print(f"A sqlite3.Error occurred while executing SQL: {e}")
    else: # if len(new_documents_sql) > 0:
        safe_print(f"\t- inserting {len(new_documents_sql)} new documents")
        cursor = conn.cursor()
        for sql in new_documents_sql:
            try:
                # Split the SQL into the base query and the values
                
                base_sql = sql.split('VALUES', 1)[0] + 'VALUES'
                values = sql.split('VALUES', 1)[1].strip('()')
                
                if DEBUG_MODE:
                    safe_print(f"{sql}")
                    safe_print(f"{base_sql}")
                    safe_print(f"{values}")
                
                # Parse the values from the SQL string
                value_list = values.split(',')[:5]
                params = []
                for value in value_list:
                    # Strip any quotes and whitespace
                    value = value.strip("' ").strip('"')
                    params.append(value)
                
                if DEBUG_MODE:
                    safe_print(f"{params}")
                
                # Execute the SQL with parameters
                cursor.execute(base_sql + ' (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)', params)
            except sqlite3.Error as e:
                safe_print(f"A sqlite3.Error occurred while executing SQL {sql}, {e}")
        conn.commit()
        
    if update_documents_sql:
        if len(update_documents_sql) > 0:
            safe_print(f"\t- updating {len(update_documents_sql)} existing documents")
            cursor = conn.cursor()
            for sql in update_documents_sql:
                try:
                    cursor.execute(sql)
                    
                except sqlite3.Error as e:
                    safe_print(f"A sqlite3.Error occurred while executing SQL {sql}, {e}")
            conn.commit()
    
    safe_print(f"\t- {doc_href_total} linked documents were evaluated.")
    
    end_time = datetime.now()
    safe_print(f"... data handling finished, time elapsed {end_time-start_time_db}.")

import concurrent.futures
import requests

driver_lock = Lock()

def handle_request_pages_threaded(conn, query=None, nodoc=False, workers=14, batch_size=25, staging=False):
    global doc_href_total, doc_href_inserted, total, nr, big_total, doc_href_updated
    cursor = conn.cursor()
    
    try:
        cursor.execute(query)
    except sqlite3.Error as e:
        print(f"SQLite error executing query: {e}")
        conn.rollback()
        return
    
    rows = cursor.fetchall()
    total = len(rows)
    
    results = return_all_registered_documents(conn)
    all_documents = results[0]
    documents_set = results[1]
    
    # all_documents = {doc[0]: doc for doc in cursor.fetchall()}
    # documents_set = set(doc[0] for doc in all_documents.values())
    
    grand_total_inserted = 0
    grand_total_updated = 0
    doc_href_total = 0
    doc_href_inserted = 0
    doc_href_updated = 0
    big_total = 0
    workers = int(workers)
    batch_size = int(batch_size)
    
    document_references = []
    request_tags = []
    detailed_tags = []
    sql_results = []
    new_documents_sql = []
    existing_documents_sql = []
    
    safe_print(f"Number of records returned from query {total}, total registered documents {len(all_documents)}", force=True)
    
    total_unique_docs = total
    number_of_batches = calculate_batches(total_unique_docs, batch_size)
    last_batch = total_unique_docs - ((number_of_batches-1) * batch_size)
    batch_number = 1
    nr = 0
    
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    stop_file = f"{script_name}.stop"

    if os.path.exists(stop_file):
        stopped = True
        safe_print(f"Stop file '{stop_file}' detected. Stopping the evaluation.", force=True)
        return

    # Use a pool of drivers to avoid creating new instances for each URL
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
    chrome_options.add_argument("--mute-audio")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--start-minimized")
    chrome_options.add_argument("--log-level=3")

    try:
        driver_pool = [webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options) for _ in range(workers)]
        # driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    except Exception as e:
        print(f"ChromeDriver initialization failed: {e}")
        return 
        

    def fetch_page_threaded(url, driver):
        try:
            driver.get(url)
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            if VERBOSE_MODE:
                safe_print(f"fetch_page_threaded: driver.get {url}")
                
            return soup
        except (TimeoutException, WebDriverException) as e:
            try:
                driver.get(url)
                WebDriverWait(driver, 10).until(lambda d: d.find_element(By.TAG_NAME, 'body'))
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                return soup
            except (TimeoutException, WebDriverException) as e:
                safe_print(f"Error fetching page {url}: {e}")
                return None

    def process_row(row, driver_index, doc_nr):
        global doc_href_total, doc_href_inserted, nr
        inserted_recs = 0
        updated_recs = 0
        sql_return = []
        session = None
        
        try:
            title, dossier_url = row
            nr += 1
            safe_print(f"({nr}/{total}) Fetching {dossier_url}")
        except ValueError:
            print(f"Error: Unexpected row structure for {row}")
            return 0, 0, []

        try:
            with driver_lock:
                driver = driver_pool[driver_index]
            # detailed_soup = fetch_page_threaded(dossier_url, driver)
            detailed_soup = fetch_page(dossier_url, driver)
            if detailed_soup is None:
                return 0, 0, []

            page_count = 1
            detailed_tags = extract_all_tags_from_soup(detailed_soup[1], dossier_url, session, datetime.now().strftime('%Y-%m-%d'), VERBOSE_MODE, nodoc, all_documents, documents_set, doc_nr, page_count)
            
            # return tags, document_references, sql_results
            doc_href_total += len(document_references)
            # if not nodoc or test:
                # results = extract_and_process_links(str(detailed_soup), dossier_url, 'requests_tags', list(all_documents.values()), documents_set, conn, nodoc, False)
                # doc_href_total += results[0]
                # doc_href_inserted += results[1]
            # safe_print("detailed_tags",detailed_tags)
            
            msg = ""
            
            if detailed_tags[0]:
                msg += f"{len(detailed_tags[0])} added to {len(request_tags)}\n"
                request_tags.extend(detailed_tags[0])
                
            if detailed_tags[1]:
                msg += f"{len(detailed_tags[1])} added to {len(document_references)}\n"
                # for doc in detailed_tags[1]:
                document_references.append(detailed_tags[1])
                
            if detailed_tags[2]:
                msg += f"{len(detailed_tags[2])} added to {len(sql_results)}\n"
                sql_results.extend(detailed_tags[2])
                
            if True:
                safe_print(msg)
            
            for tag in request_tags:
                try:
                    result = update_or_insert_tag(conn, 'requests_tags', tag, staging, VERBOSE_MODE, True)
                    if result:
                        inserted_recs += result[0]
                        updated_recs += result[1]
                        if result[0] > 0 or result[1] > 0:
                            sql_return.append(result[2])
                except sqlite3.Error as e:
                    print(f"SQLite error while processing tag {tag['tag_key']}: {e}")
                    conn.rollback()
                except Exception as e:
                    safe_print(f"Unexpected error while processing tag {tag['tag_key']}: {e}")
                    conn.rollback()
            
            doc_list = []
            
            # for document in document_references:
                # n = 0
                # href_list = [row[-1] for row in document]
                # for record in document:
                    # if n > 0:
                        # doc_list.append(record)
                    # else:
                        # n += 1
                        
            href_list = [row[-1] for row in document_references]
                
            if DEBUG_MODE:
                safe_print(f"href_list {len(href_list)}")
            
            if href_list:
                
                safe_print(f"new_docs = process_documents({len(doc_list)}, {len(documents_set)}, dossier_url)")
                new_docs, existing_docs = process_documents(document_references, documents_set, dossier_url)
            else:
                new_docs = []

            if new_docs:
                if DEBUG_MODE:
                    safe_print('len(new_docs)', len(new_docs))# Insert documents
                for document in new_docs:
                    if document:  # Ensure document isn't None
                        if isinstance(document, dict):
                            if DEBUG_MODE:
                                safe_print(f"Document keys: {list(document.keys())[:5]}")  # Print first 5 keys
                        elif isinstance(document, str):
                            if DEBUG_MODE:
                                safe_print(f"Document content: {document[:100]}")
                        else:
                            if DEBUG_MODE:
                                safe_print(f"Document type: {type(document)}")

                        try:
                            if document.get('Locatie open.minvws.nl') != 'URL':  # Use .get() for safety
                                insert_sql = """
                                INSERT INTO documents 
                                (Id, [Locatie open.minvws.nl], [Document naam], FromFile, [Document ID], ModifiedDate, CreatedDate) 
                                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                                """
                                formatted_sql = insert_sql.replace("?", "'{}'").format(document['Id'], document['Locatie open.minvws.nl'], document['Document naam'], document['FromFile'], document['Document ID'])
                                new_documents_sql.append(formatted_sql)
                                doc_href_inserted += 1
                        except KeyError as ke:
                            safe_print(f"\tKeyError occurred: {ke}. Document might be missing required fields.")
                        except Exception as e:
                            safe_print(f"\tAn exception occurred gathering SQL statements: {e}")
            
            if new_documents_sql:
                if len(new_documents_sql) > 0:
                    safe_print(f"\t- inserting {len(new_documents_sql)} new documents")
                    cursor = conn.cursor()
                    for sql in new_documents_sql:
                        try:
                            # Split the SQL into the base query and the values
                            
                            base_sql = sql.split('VALUES', 1)[0] + 'VALUES'
                            values = sql.split('VALUES', 1)[1].strip('()')
                            
                            if DEBUG_MODE:
                                safe_print(f"{sql}")
                                safe_print(f"{base_sql}")
                                safe_print(f"{values}")
                            
                            # Parse the values from the SQL string
                            value_list = values.split(',')[:5]
                            params = []
                            for value in value_list:
                                # Strip any quotes and whitespace
                                value = value.strip("' ").strip('"')
                                params.append(value)
                            
                            if DEBUG_MODE:
                                safe_print(f"{params}")
                            
                            # Execute the SQL with parameters
                            cursor.execute(base_sql + ' (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)', params)
                        except sqlite3.Error as e:
                            safe_print(f"A sqlite3.Error occurred while executing SQL {sql}, {e}")
                    
                    conn.commit()
            
            if existing_docs:
                doc_href_updated = 0
                for document in existing_docs:
                    if document:  # Ensure document isn't None
                        update_sql = """UPDATE documents SET ModifiedDate = CURRENT_TIMESTAMP WHERE [Locatie open.minvws.nl] = ?"""
                        try:
                            formatted_sql = update_sql.replace("?", "'{}'").format(document['Locatie open.minvws.nl'])
                            existing_documents_sql.append(formatted_sql)
                            doc_href_updated += 1
                            
                        except sqlite3.Error as e:
                            safe_print(f"\tDatabase error while updating document: {e}")
            else:
                if VERBOSE_MODE:
                    safe_print(f"process_documents returned no existing docs for {len(existing_docs)} references")   
                    
            if sql_return:
                # def save_results(results, conn, verbose, number_of_batches, batch_number, batch_size, last_result, the_worker, staging):
                save_results(sql_return, conn, VERBOSE_MODE, number_of_batches, batch_number, len(sql_return), last_batch, driver_index, staging)
                return inserted_recs, updated_recs, sql_return, doc_href_updated
                
            else:
                return inserted_recs, updated_recs, [], doc_href_updated
                
        except requests.RequestException as e:
            safe_print(f"Failed to fetch detailed page for {title}, {dossier_url}: {e}")
        except Exception as e:
            safe_print(f"Error processing {dossier_url}: {e}")
        
        return 0, 0, [], 0

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:  
            futures = [executor.submit(process_row, row, i % len(driver_pool), i) for i, row in enumerate(rows)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    inserted, updated, sql_return, nr_docs = future.result()
                    grand_total_inserted += inserted
                    grand_total_updated += updated
                    doc_href_total += nr_docs
                    
                except Exception as e:
                    print(f"An error occurred in thread: {e}")
                
                if os.path.exists(stop_file):
                    safe_print(f"Stop file '{stop_file}' detected. Stopping the evaluation.", force=True)
                    for f in futures:
                        f.cancel()
                    break
    finally:
        # Ensure all drivers are closed even if an exception occurs
        for driver in driver_pool:
            try:
                driver.quit()
            except:
                pass

    safe_print(f"\nFor {total} results {grand_total_inserted} new tags were found and {grand_total_updated} existing tags were updated. {doc_href_total} linked documents were evaluated of which " + ("none weren't already registered." if doc_href_inserted == 0 else f"{doc_href_inserted} were added to the database."))
    
# ##################################################### #
# Use browser emulation to retrieve pages. Not perfect. #
# How much not perfect status currently unknown :)      # 
# Might fail. Difficult on Synology (fwiw). Windows 11  #
# might show selenium window.                           #
# ----------------------------------------------------- #
# Returns file_name, new_download_url                   #
# ##################################################### #
# caching headless browser
@functools.lru_cache(maxsize=5000)  # up from 100
def cached_selenium_operation(url, use_firefox=True, firefox_path=None):
    global driver

    if driver is None:
        if use_firefox:
            firefox_options = FirefoxOptions()
            firefox_options.add_argument("--headless")
            if firefox_path:
                firefox_options.binary_location = firefox_path
            driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=firefox_options)
        else:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
            chrome_options.add_argument("--mute-audio")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--start-minimized")
            chrome_options.add_argument("--log-level=3")
            
            safe_print(f"ChromeDriver Installation Path: {ChromeDriverManager().install()}")
            try:
                driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
            except Exception as e:
                print(f"ChromeDriver initialization failed: {e}")
                if use_firefox:
                    # If Chrome fails and use_firefox is True, fallback to Firefox
                    firefox_options = FirefoxOptions()
                    firefox_options.add_argument("--headless")
                    if firefox_path:
                        firefox_options.binary_location = firefox_path
                    driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=firefox_options)

    try:
        if driver:
            driver.get(url)
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "a.woo-button.woo-button--primary")))
            if driver:
                new_link = driver.find_element(By.CSS_SELECTOR, "a.woo-button.woo-button--primary")
            if new_link:
                new_download_url = new_link.get_attribute('href')
            if new_download_url:
                new_response = requests.head(new_download_url, allow_redirects=True)
            if 'Content-Disposition' in new_response.headers:
                content_disposition = new_response.headers['Content-Disposition']
                filename_matches = re.findall(r'filename[^;=\n]*=(([\'"]).*?\2|[^;\n]*)', content_disposition)
                if filename_matches:
                    file_name = filename_matches[0][0].strip('"')
                    return file_name, new_download_url

        # If no Content-Disposition, try to extract from the URL itself
        file_name = urlparse(new_download_url).path.split('/')[-1]
        return file_name, new_download_url
    except Exception as e:
        print(f"Error processing URL {url}: {str(e)}")
        return None, url

# ############################################### #
# Used when switching browsers for handling pages #
# ----------------------------------------------- #
# Returns Windows 11 path to Firefox              #
#         Linux path to Firefox                   #
# ############################################### #
def get_firefox_path():
    # Detect OS
    if platform.system() == "Windows":
        # Assuming a default installation path for Firefox on Windows
        return r"C:\Program Files\Mozilla Firefox\firefox.exe"
    elif platform.system() == "Linux":
        # Replace this with the correct path on your Linux system
        return "/mysysenv/firefox/firefox"
    else:
        print("Unsupported operating system")
        return None

# #################################################### #
# Files are downloaded with the file_name as presented #
# by the host. What file_name that is is handled in    #
# this function. Follows redirects.                    #
# ---------------------------------------------------- #
# Uses cached_selenium_operation                       #
#      get_firefox_path                                #
# ---------------------------------------------------- #
# Returns file_name, url                               #
# #################################################### #
def extract_filename_and_url_from_headers(url_or_response):
    """
    Extract the filename from the response headers for 'document' or 'bijlage' URLs using a minimal Selenium setup.
    Can handle both URL strings and Response objects. Also checks for 404 responses.

    :param url_or_response: URL string or Response object to process
    :return: A tuple of (filename, download_url) or raises an exception if unable to determine
    """
    url = url_or_response.url if hasattr(url_or_response, 'url') else url_or_response
    if DEBUG_MODE:
        safe_print(f"Debug: URL being processed: {url}")

    # First, try to get the filename from headers without using Selenium
    if hasattr(url_or_response, 'headers'):
        response = url_or_response
    else:
        try:
            response = requests.head(url, allow_redirects=True, timeout=10)  # Use HEAD for efficiency, set timeout
        except requests.RequestException as e:
            if VERBOSE_MODE:
                safe_print(f"Debug: Request failed with exception: {e}")
            raise Exception(f"Failed to fetch headers for URL {url}: {str(e)}")

    if DEBUG_MODE:
        print("Debug: ", response.headers)
    
    # Check for 404 status code
    if response.status_code == 404:
        if VERBOSE_MODE:
            safe_print(f"Debug: 404 Not Found for URL: {url}")
        return '404', url
        # raise Exception(f"404 Not Found for URL: {url}")

    if 'Content-Disposition' in response.headers:
        content_disposition = response.headers['Content-Disposition']
        filename_matches = re.findall(r'filename[^;=\n]*=(([\'"]).*?\2|[^;\n]*)', content_disposition)
        if filename_matches:
            file_name = filename_matches[0][0].strip('"')
            if DEBUG_MODE:
                safe_print(f"Debug: Filename found in headers: {file_name}")
            if file_name.lower() != 'document':
                return file_name, url
        
        filename_star_match = re.search(r'filename\*=UTF-8\'\'(.*)', content_disposition)
        if filename_star_match:
            file_name = unquote(filename_star_match.group(1))
            if DEBUG_MODE:
                safe_print(f"Debug: Filename* found in headers: {file_name}")
            if file_name.lower() != 'document':
                return file_name, url

    # If all else fails, try to guess filename from URL
    parsed_url = urlparse(url)
    path = parsed_url.path
    file_name = path.split('/')[-1] if '/' in path else None
    if file_name == None:
        if DEBUG_MODE:
            safe_print(f"Debug: Filename None: {file_name}")
        return None, url

    # If no filename found or if the URL didn't match any of the patterns, raise an exception
    return None, url
    # raise Exception(f"Filename {file_name} could not be determined, or new URL {url} could not be found.")

# ################################# #
# Download a file with retry logic  #
# --------------------------------- #
# Returns True on Succes else False #
# ################################# #
def download_file(url, local_path):
    """Download a file with retry logic for network issues."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        safe_print(f"download_file: saving into {local_path}")
        with open(local_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if stop_threads:
                    safe_print(f"Stopping download for {url}")
                    file.close()
                    os.remove(local_path)
                    return False
                file.write(chunk)
        if VERBOSE_MODE:
            safe_print(f"download_file: successfully downloaded")
        return True
    except requests.RequestException as e:
        if VERBOSE_MODE:
            safe_print(f"download_file: failed to download {url}: {e}")
        return False

# ############################################ #
# Uses definition from wook.config, attribute  #
# URLkeys. The system queries the database for #
# the links that are defined with this value.  #
# It could be "DocumentDownloadLink",          #
# "Aanvullende bijlagen_link_1",               #
# "Omvang openbaarmaking_link_1" etc..         #
# After retrieving URL tries to download tile  #
# into the designated folder as defined by     #
# DOWNLOAD_INVENTARIS, DOWNLOAD_BESLUIT and    #
# DOWNLOAD_DOCUMENTEN                          #
# -------------------------------------------- #
# Uses download_file                           #
#      "URLkeys" from table config             #
# Returns Updated/Inserted records in either   #
#         requests_tags or documents_tags      #
# Command line: --download                     # 
# ############################################ #
def process_row(row, download_path, force):

    global stop_threads

    while not stop_threads:
        title, tag_key, download_url, bestandsnaam = row
        # Check if bestandsnaam has an extension
        # safe_print('bestandsnaam', bestandsnaam)
        if bestandsnaam and bestandsnaam != '--':
            if not os.path.splitext(bestandsnaam)[1]:  # If the extension is empty
                bestandsnaam += '.pdf'
                safe_print('bestandsnaam', bestandsnaam)
            
        i = b = d = 0
        try:
            if ("besluitbrief_link" in tag_key.lower() or 
                "aanvullende bijlagen_link" in tag_key.lower() or 
                "besluitbrief_actual_download_url" in tag_key.lower()):
                b += 1
                # save_folder = DOWNLOAD_BESLUIT 
            elif "documentdownload_linkhref" in tag_key.lower():
                d += 1
                # save_folder = DOWNLOAD_DOCUMENTEN
            elif "documentdownload_proposed_filename" in tag_key.lower():
                d += 1
                # save_folder = DOWNLOAD_DOCUMENTEN
            elif "documentdownloadlink" in tag_key.lower(): 
                d += 1
                # save_folder = DOWNLOAD_DOCUMENTEN
            elif "download document archief link_1_href" in tag_key.lower():
                b += 1
                # save_folder = DOWNLOAD_BESLUIT
            elif download_url.endswith('download/inventory'):
                # save_folder = DOWNLOAD_INVENTARIS
                i += 1
            elif "omvang openbaarmaking_" in tag_key.lower():
                # save_folder = DOWNLOAD_INVENTARIS
                i += 1
            else:
                b += 1
                # save_folder = DOWNLOAD_BESLUIT
            
            full_url = urljoin(base_url, download_url) if not download_url.startswith('http') else download_url
            message = ""
            
            if bestandsnaam and bestandsnaam != '--':
                local_path = os.path.join(download_path, bestandsnaam)  #!
                if DEBUG_MODE:
                    safe_print('local_path',local_path)
                file_exists = os.path.exists(local_path)
                if VERBOSE_MODE:
                    safe_print(f"process_row: retrieved {title}, {tag_key}, {full_url}, {bestandsnaam} for download into {local_path})")
                if force or not file_exists:
                    if VERBOSE_MODE:
                        safe_print(f"process_row: file {bestandsnaam} does NOT exist at {local_path}...")
                    file_exists = False
                else:
                    if VERBOSE_MODE:
                        safe_print(f"process_row: file {bestandsnaam} exists on {local_path}... skip to next")
                    return i, b, d
            elif bestandsnaam == '--':
                try:
                    bestandsnaam, download_url = extract_filename_and_url_from_headers(requests.head(full_url, allow_redirects=True))
                    if bestandsnaam == '404':
                        safe_print(f"process_row: 404 on url {download_url}")
                        return i, b, d
                    elif bestandsnaam and download_url and bestandsnaam != '404':
                        full_url = urljoin(base_url, download_url) if not download_url.startswith('http') else download_url
                        local_path = os.path.join(download_path, bestandsnaam)
                        file_exists = os.path.exists(local_path)
                        if DEBUG_MODE:
                            safe_print(f"process_row: retrieved {bestandsnaam} from {download_url}")
                    elif not bestandsnaam and download_url:
                        file_exists = False
                        full_url = urljoin(base_url, download_url) if not download_url.startswith('http') else download_url
                        bestandsnaam = ""
                        local_path = os.path.join(download_path, bestandsnaam)
                    else:
                        if VERBOSE_MODE:
                            safe_print(f"process_row: file '{bestandsnaam}' and url '{download_url}'")
                        
                except Exception as e:
                    if '404 Not Found' in str(e):
                        safe_print(f"'404 Not found' error for {full_url}")
                    else:
                        safe_print(f"'{e}' error for {full_url}")
                    return i, b, d
                
            try:
                if not file_exists:
                    if download_file(full_url, local_path):
                        safe_print(f"process_row: file {bestandsnaam} downloaded successfully.")
                    if os.path.exists(local_path):
                        safe_print(f"process_row: file {bestandsnaam} exists")
                    else:
                        safe_print(f"process_row: file {bestandsnaam} does not exist")
                else:
                    if VERBOSE_MODE:
                        safe_print(f"process_row: file already exists.")
                
            except Exception as e:
                safe_print(f"process_row: exception, {e}")
        
        except KeyboardInterrupt:
            safe_print("Thread interrupted. Exiting process_row.")
            return i, b, d  # Return current counts even if interrupted
        
        return i, b, d
        
        if stop_threads:
            return i, b, d  

def download_files_from_tags(conn, download_path, query, max_files=None, force=False, workers=4):
    global stop_threads
    cursor = conn.cursor()
    
    stop_threads = False
    
    if workers:
        workers = int(workers)
        
    if not query:
        safe_print(f"download_files_from_tags: download function needs query to tag with URL")
        return 0, 0, 0
    else:
        safe_print(f"download_files_from_tags: downloading files using query {query}")
        
        cursor.execute(query)
        rows = cursor.fetchall()
        safe_print(f"download_files_from_tags: found {len(rows)} records for query for {download_path}.")
        total_rows = len(rows)

    def signal_handler(signum, frame):
        safe_print("Caught interrupt signal. Exiting...")
        raise KeyboardInterrupt

    if total_rows == 0:
        safe_print("No results from query. Nothing to do.")
        return 0, 0, 0
        
    # Set the signal handler for SIGINT (CTRL+C)
    signal.signal(signal.SIGINT, signal_handler)

    # Get the set of full paths from the query results
    query_filenames = {row[3] for row in rows}  # Now 'bestandsnaam' contains subfolder path

    # Extract unique path prefixes from query results
    unique_prefixes = {os.path.dirname(path) for path in query_filenames if path != '--' and path != ''}
    
    # Collect all files only from the directories specified in unique_prefixes
    files_in_path = set()
    safe_print(f"download_files_from_tags: unique_prefixes {unique_prefixes}")
    for prefix in unique_prefixes:
        full_path = os.path.join(download_path, prefix)
        old_value = len(files_in_path)
        safe_print(f"download_files_from_tags: handling {full_path} and adding files.")
        if os.path.exists(full_path):
            for file in os.listdir(full_path):
                relative_path = os.path.join(prefix, file)
                files_in_path.add(relative_path)
        safe_print(f"download_files_from_tags: added {len(files_in_path)-old_value} files from {full_path}.")

    safe_print(f"download_files_from_tags: in total found {len(files_in_path)} files for root {download_path}.")
    safe_print("")
    
    # Find files in query that are not in download_path
    files_to_process = query_filenames - files_in_path

    # Convert filenames to rows for processing, adjusting for the correct folder
    rows_to_process = [row for row in rows if row[3] in files_to_process]
    
    safe_print(f"download_files_from_tags: found {len(rows_to_process)} files to process.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_row = {executor.submit(process_row, row, download_path, force): row for row in rows_to_process[:max_files if max_files else len(rows_to_process)]}
        i_total, b_total, d_total = 0, 0, 0
        try:
            for future in concurrent.futures.as_completed(future_to_row):
                if stop_threads:
                    executor.shutdown(wait=False)
                    break
                try:
                    i, b, d = future.result()
                    i_total += i
                    b_total += b
                    d_total += d
                except KeyboardInterrupt:
                    executor.shutdown(wait=False)
                    safe_print("KeyboardInterrupt caught in executor. Gathering completed results and then exiting.")
                    break
                except Exception as e:
                    safe_print(f"Error in thread: {e}")
        except KeyboardInterrupt:
            executor.shutdown(wait=False)
            safe_print("KeyboardInterrupt caught. Exiting.")
            return i_total, b_total, d_total

    return i_total, b_total, d_total
    
# ####################################### #
# Scrape search page open.minvws.nl       #
# Adds found attributes to requests_tags. # (adds attributes to _tags that are also added to requests)
# --------------------------------------- #
# Returns data object with results from   #
#         page                            # 
# ####################################### #
def get_search_results(soup, staging, title_tags, base_href='', conn=None, VERBOSE_MODE=False):
    """
    Extract result information from the BeautifulSoup object and handle various formats.

    :param soup: BeautifulSoup object
    :param base_href: Base URL for relative links
    :param conn: Database connection object
    :param VERBOSE_MODE: Boolean flag to enable debug output
    :return: List of dictionaries containing result information
    """
    if soup:
        results = soup.find(id="search-results-list")   # ul
    data = []
    sql_return = []
    inserted_recs = 0
    updated_recs = 0
            
    if results:
        entries = results.find_all('li', class_='woo-search-result')
        nr = 0
        if VERBOSE_MODE:
            safe_print(f"\t\t- on this page {len(entries)} results are found")
        
        for result in entries:  # every results is handled seperatly
            record = {}
            header = result.find('header', class_='woo-search-result__header')
            if not header:
                continue
 
            # Extract Title and DossierURL
            # into requests.<Column name>
            title_elem = header.find('a', class_='woo-search-result__main-link')
            if title_elem:
                nr += 1
                record['Title'] = title_elem.text.strip()
                record['DossierURL'] = urljoin(base_href, title_elem['href'])
                if VERBOSE_MODE:
                    print(f"{nr} Debug: Title: {record['Title']}")
                    print(f"{nr} Debug: Dossier URL: {record['DossierURL']}")

            # Extract from the first spec list
            # into requests.<Column name>
            first_spec_list = header.find('ul', class_='woo-search-result__spec-list')
            if first_spec_list:
                spec_items = first_spec_list.find_all('li', class_='woo-search-result__spec')
                if len(spec_items) > 0:
                    record['DecisionType'] = spec_items[0].find('span', class_='font-bold').text if spec_items[0].find('span', class_='font-bold') else ""
                if len(spec_items) > 1:
                    document_count_text = spec_items[1].text.strip()
                    record['DocumentCount'] = ''.join(filter(str.isdigit, document_count_text)) or "0"
                    record['DocumentCountOri'] = document_count_text  # Store original text for reference
                if len(spec_items) > 2:
                    record['DisclosureType'] = spec_items[2].text.strip()

            # Extract from the second spec list
            # into requests.<Column name>
            second_spec_list = result.find_all('ul', class_='woo-search-result__spec-list')
            if second_spec_list:
                for spec in second_spec_list[-1].find_all('li', class_='woo-search-result__spec'):
                    text = spec.text.strip()
                    if 'Besluit genomen op' in text:
                        date_str = re.search(r'(\d{1,2} \w+ \d{4})', text)
                        if date_str:
                            try:
                                date = datetime.strptime(date_str.group(1), '%d %B %Y').strftime('%Y-%m-%d')
                                record['DecisionDate'] = date
                            except ValueError:
                                pass  # Handle any date parsing errors silently for now
                    elif 'gepubliceerd op' in text:
                        pub_date_elem = spec.find('time', datetime=True)
                        if pub_date_elem:
                            record['PublicationDate'] = pub_date_elem['datetime']
                    else:
                        record['DossierNumber'] = text

            # Database insertion with the new logic
            if conn and record:
                cursor = conn.cursor()
                table_config = next((config for config in TABLE_CONFIGS if config['name'] == 'requests_tags'), None)
                if table_config:
                    for tag_key, tag_value in record.items():
                        try:
                            # Ensure tag_value is a string to prevent SQL injection and ensure consistency
                            tag_value = str(tag_value)
                            tag_value = re.sub(r'\s+', ' ', tag_value).strip()
                            
                            # Correctly call update_or_insert_tag with individual tag values
                            # def update_or_insert_tag(conn, table_name, record, staging, verbose=False, in_batch=False, existing_tag_values=None):
                            #                         result = update_or_insert_tag(conn, 'requests_tags', tag, staging, VERBOSE_MODE, in_batch, existing_tag_values=title_tags[dossier_url])

                            
                            result = update_or_insert_tag(conn, 'requests_tags', 
                                                                  {'Title': record.get('DossierURL', ''), 'tag_key': tag_key, 'tag_value': tag_value},
                                                                  staging,
                                                                  VERBOSE_MODE,
                                                                  False,
                                                                  existing_tag_values=title_tags.get(record.get('DossierURL', ''), []))
                            if result[0] > 0 or result[1] > 0:
                                inserted_recs += result[0] # inserted
                                updated_recs += result[1] # updated
                                sql_return.append(result[2]) # sql statements
                                # print(f"\t- retrieved {inserted_recs} records for insert, {updated_recs} for update")
                                # sql_return.extend(result[2])
                        except sqlite3.Error as e:
                            if VERBOSE_MODE:
                                print(f"An error occurred while inserting/updating {record['Title']} tag {tag_key}: {e}")

            data.append(record)
        
    return data, inserted_recs, updated_recs, sql_return
    
# ################################ #
# Rate limit the requests, be nice #
# ################################ #
def rate_limited_call(func, *args, **kwargs):
    """A decorator to enforce rate limit."""
    time.sleep(2)  # Adjust based on the rate limit policy of the server
    return func(*args, **kwargs)

# help text #
def about_message():
    about_text = f"""
    versie {SCRIPT_VERSION}
    functie parameters script:
    --db <database name>        verplicht, wordt aangemaakt sqlite3
    --search                    doorzoek de open.minvw.nl dossiers zoekpagina
    --requests                  evalueer de geregistreerde dossiers
    --download                  downloaden archiven en geregistreerde documenten
    
    sql parameters:
    --query <sql query>         gebruik deze query als input voor de functie
    --limit <nr>                limiteer het aantal te evalueren records
   
    metrics:
    op een Intel(R) Core(TM) Ultra 9 185H, 2300 Mhz, 16 Core(s) met 22 Logical Processor(s),
    64GB RAM en een Model KXG8AZNV1T02 LA KIOXIA SSD, duurt stap 1) --search ongeveer een
    2 minuten. stap 2) --requests ongeveer een uur, stap 3) afhankelijk van de
    bandbreedte.
    
    """
    print(about_text)

# ############################### #
# Main function, handle arguments #
# ############################### #
def main():
    global VERBOSE_MODE
    global DEBUG_MODE
    global args
    global conn
    global driver
    global start_batch_time
    global SILENT_MODE
    global start_time
    global FORCE
    global FORCE_MODE
    global WRITE_RESULTS
    
    # Create parser
    parser = argparse.ArgumentParser(
        description=f"wookey.py {SCRIPT_VERSION} - process Woo requests and documents from open.minvws.nl",
        add_help=False  # We'll handle help ourselves
    )
    parser.add_argument('-h', '--help', action='help', help="Show this help message and exit.")
    parser.add_argument('--about', action='store_true', help="Show information about the program's development.")
    parser.add_argument('--db', required=True, help="SQLite database file name")

    parser.add_argument('--download', metavar='download_path', nargs='?', const='.', default=None, help="Path to download files. If path contains spaces, enclose it in quotes.")
    parser.add_argument('--force', action='store_true', help="Force updates even if file exists locally.")
    parser.add_argument('--files', type=int, default=0, help="Number of files to download. Use 0 or omit to download all files.")
    parser.add_argument('--table', metavar='tags', nargs='?', const='requests_tags', default=None, help="Table with download tags.")
    parser.add_argument('--tags', metavar='tags', nargs='?', const='"Download document archief Link_1_href"', default=None, help="Tags with download links.")
    
    parser.add_argument('--debug', action='store_true', help="Print DEBUG messages to screen.")
    
    parser.add_argument('--verbose', action='store_true', help="Print all messages including debug to screen.")
    
    parser.add_argument('--silent', action='store_true', help="Suppress all messages.")

    parser.add_argument('--staging', action='store_true', help="Store results in staging table.")
    
    parser.add_argument('--search', action='store_true', help="Evaluate the search page of open.minvws.nl and save the results in the requests_tags table.")
    
    parser.add_argument('--requests', action='store_true', help="Evaluate the registered DossierURL's from requests_tags.")    
    parser.add_argument("--eval", help="Evaluate new document URL's", action="store_true")
    parser.add_argument("--period", default="lastday", help="Evaluate URL's from a certain period ")

    parser.add_argument('--nodoc', action='store_true', help="When checking requests pages do not check listed documents.")
    parser.add_argument('--nowritetocache', action='store_true', help="Do not create local cache.")
    parser.add_argument('--query', metavar='SQL_QUERY', nargs='?', default=None, help="Custom SQL query to be used for the selected function.")
    parser.add_argument('--limit', metavar='LIMIT_QUERY', nargs='?', default=None, help="Limit to <nr> results.")
    
    parser.add_argument('--thread', action='store_true', help="Use threading.")
    parser.add_argument('--wait', metavar='WAIT_MILLISEC', nargs='?', default=None, help="Milliseconds to wait between workers.")
    parser.add_argument("--batch", metavar='BATCH_SIZE', nargs='?', default=32, help="Batch size for saving results (def. 32) ")
    parser.add_argument("--workers", metavar='WORKERS', nargs='?', default=3, help="Workers for parallel processing (def. 3) ")
    
    # welcome
    print(f"Starting script: {os.path.basename(__file__)} to process Woo documents from open.minvws.nl.")
    
    # patch
    use_firefox = False

    # Check if no arguments or help flag is provided
    if len(sys.argv) == 1 or '--help' in sys.argv or '-h' in sys.argv:
        parser.print_help()
        return

    args, unknown = parser.parse_known_args()
    
    VERBOSE_MODE = args.verbose

    if args.about:
        about_message()
        return
    
    if not args.nowritetocache:
        WRITE_RESULTS = True
    else:
        WRITE_RESULTS = False
        
    if not args.batch:
        BATCH_SIZE = 32
    else:
        BATCH_SIZE = args.batch
    
    if not args.workers:
        WORKERS = 3
    else:
        WORKERS = args.workers

    if not args.thread:
        THREAD = False
    else:
        THREAD = True
    
    if not args.nodoc:
        NODOC = False
    else:
        NODOC = True

    if not args.wait:
        WAIT_MILLISEC = 10
    else:
        WAIT_MILLISEC = int(args.wait)
    
    if not args.force:
        FORCE = False
        FORCE_MODE = False
    else:
        FORCE = True
        FORCE_MODE = True
    
    if not args.debug:
        DEBUG_MODE = False
    else:
        DEBUG_MODE = True
        VERBOSE_MODE = True
    
    # print(f"{args.trace} {args.verbose}")
    
    if not args.staging:
        STAGING = False
    else:
        STAGING = True
        FORCE = True
        
    if args.silent:
        script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        stop_file = f"{script_name}.stop"
        print(f"Switching to silent mode. When saving results totals are shown. Brake? Create {stop_file}.")
        SILENT_MODE = True
    elif not args.silent:
        SILENT_MODE = False
        
    
                
    all_results = []
    
    conn = create_connection(args.db, TABLE_CONFIGS[0].get('db_platform', 'sqlite3'))
    ensure_tables_exist(conn, TABLE_CONFIGS)
    create_folders()

    if args.limit:
        LIMIT_QUERY = f"LIMIT {args.limit}"
    else:
        LIMIT_QUERY = ""
    
    # DEFAULT SQL QUERIES
    if not args.query:
        if args.requests:
            # query = f"SELECT DISTINCT rt.Title, rt.tag_value FROM requests_tags rt join requests_tags rt2 ON rt.Title = rt2.Title and rt2.tag_key LIKE 'ContainsDocument_na_linkdate_100' WHERE rt.tag_key = 'DossierURL' ORDER BY RANDOM() {LIMIT_QUERY}"
            query = f"SELECT DISTINCT rt.Title, rt.tag_value FROM requests_tags rt WHERE rt.tag_key = 'DossierURL' ORDER BY RANDOM() {LIMIT_QUERY}"
    else:
        query = args.query + " " + LIMIT_QUERY
        
    
    if args.db:
        if args.eval:
            
            # Ensure table configurations for documents_tags and requests_tags are loaded
            doc_tags_config = next((table for table in TABLE_CONFIGS if table['name'] == 'documents_tags'), None)
            req_tags_config = next((table for table in TABLE_CONFIGS if table['name'] == 'requests_tags'), None)
            
            db_fields = next((table['database_fields'] for table in TABLE_CONFIGS if table['name'] == 'documents'), None)
            if not db_fields:
                print("No database fields configuration found for 'documents' table. Exiting.")
                return
                
            if not doc_tags_config or not req_tags_config:
                print("Configuration for tags tables not found in config file. Skipping evaluation.")
            else:
                if driver is None:
                    if use_firefox:
                        firefox_options = FirefoxOptions()
                        firefox_options.add_argument("--headless")
                        if firefox_path:
                            firefox_options.binary_location = firefox_path
                        driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=firefox_options)
                    else:
                        chrome_options = Options()
                        chrome_options.add_argument("--headless")
                        chrome_options.add_argument("--headless=new")
                        chrome_options.add_argument("--no-sandbox")
                        chrome_options.add_argument("--disable-dev-shm-usage")
                        chrome_options.add_argument("--disable-gpu")
                        chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
                        chrome_options.add_argument("--mute-audio")
                        chrome_options.add_argument("--disable-extensions")
                        chrome_options.add_argument("--start-minimized")
                        chrome_options.add_argument("--log-level=3")
                        
                        safe_print(f"ChromeDriver Installation Path: {ChromeDriverManager().install()}")
                        try:
                            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
                        except Exception as e:
                            print(f"ChromeDriver initialization failed: {e}")
                            if use_firefox:
                                # If Chrome fails and use_firefox is True, fallback to Firefox
                                firefox_options = FirefoxOptions()
                                firefox_options.add_argument("--headless")
                                if firefox_path:
                                    firefox_options.binary_location = firefox_path
                                driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=firefox_options)
                
                if not args.period:
                    period = "lastday"
                else:
                    period = args.period
                    
                # period
                if period == "today":
                    dateqry = "AND date(CreatedDate) = date('now')"
                elif period == "thisweek":
                    dateqry = "AND strftime('%W', CreatedDate) = strftime('%W', 'now') AND strftime('%Y', CreatedDate) = strftime('%Y', 'now');"
                elif period == "thismonth":
                    dateqry = "AND strftime('%m', CreatedDate) = strftime('%m', 'now') AND strftime('%Y', CreatedDate) = strftime('%Y', 'now');"
                elif period == "thisyear":
                    dateqry = "AND strftime('%Y', CreatedDate) = strftime('%Y', 'now');"
                elif period == "lastday":
                    dateqry = "AND date(CreatedDate) >= date('now')-1"
                elif period == "lastweek":
                    dateqry = "AND date(CreatedDate) >= date('now')-7"
                elif period == "lastmonth":
                    dateqry = "AND date(CreatedDate) >= date('now')-30"
                elif period == "lastyear":
                    dateqry = "AND date(CreatedDate) >= date('now')-365"
                else:
                    dateqry = "AND date(CreatedDate) = date('now')-1"
                
                if not args.query:
                    if FORCE:
                        query = f"""
                        SELECT DISTINCT [Locatie open.minvws.nl], [Document ID]
                        FROM documents 
                        WHERE [Locatie open.minvws.nl] NOT IN (SELECT DISTINCT Title from documents_tags)
                        AND [Locatie open.minvws.nl] IS NOT NULL AND [Locatie open.minvws.nl] != ''
                        AND CreatedDate is NOT NULL AND CreatedDate != ''
                        {dateqry}
                        ORDER BY RANDOM()
                        {LIMIT_QUERY}
                        """
                    else:
                        query_new = f"""
                        SELECT DISTINCT [Title] AS [Locatie open.minvws.nl], ID AS [Document ID] from documents_tags group by [Title] having count(*) < 24
                        {LIMIT_QUERY}
                        """
                        
                        query = f"""
                        SELECT DISTINCT [Locatie open.minvws.nl], [Document ID] 
                        FROM documents 
                        WHERE [Locatie open.minvws.nl] NOT IN 
                              (SELECT DISTINCT Title FROM documents_tags) 
                        ORDER BY CreatedDate DESC
                        {LIMIT_QUERY}
                        """
                        
                        query_old = f"""
                        SELECT DISTINCT [Locatie open.minvws.nl], [Document ID]
                        FROM documents 
                        WHERE [Locatie open.minvws.nl] IS NOT NULL AND [Locatie open.minvws.nl] != ''
                        AND CreatedDate is NOT NULL AND CreatedDate != ''
                        {dateqry}
                        ORDER BY RANDOM()
                        {LIMIT_QUERY}
                        """
                
                safe_print(f"Evaluating documents selected with query {query}")
                
                start_batch_time = datetime.now()
                start_time = datetime.now()
                
                evaluate_documents(conn, FORCE, db_fields, query, BATCH_SIZE, WORKERS, STAGING, WAIT_MILLISEC, verbose=args.verbose)
                
        elif args.download:
            print(f"Downloading files from tags ...")
            start_time = datetime.now()
            download_files_from_tags(conn, args.download, args.query, args.files, FORCE, WORKERS)
            
        elif args.requests:
            print(f"Handling request pages with query: {query}")
            print(f"DEBUG_MODE {DEBUG_MODE} VERBOSE_MODE {VERBOSE_MODE}")
            if driver is None:
                if use_firefox:
                    firefox_options = FirefoxOptions()
                    firefox_options.add_argument("--headless")
                    if firefox_path:
                        firefox_options.binary_location = firefox_path
                    driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=firefox_options)
                else:
                    chrome_options = Options()
                    chrome_options.add_argument("--headless")
                    chrome_options.add_argument("--headless=new")
                    chrome_options.add_argument("--no-sandbox")
                    chrome_options.add_argument("--disable-dev-shm-usage")
                    chrome_options.add_argument("--disable-gpu")
                    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
                    chrome_options.add_argument("--mute-audio")
                    chrome_options.add_argument("--disable-extensions")
                    chrome_options.add_argument("--start-minimized")
                    chrome_options.add_argument("--log-level=3")
                    
                    safe_print(f"ChromeDriver Installation Path: {ChromeDriverManager().install()}")
                    try:
                        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
                    except Exception as e:
                        print(f"ChromeDriver initialization failed: {e}")
                        if use_firefox:
                            # If Chrome fails and use_firefox is True, fallback to Firefox
                            firefox_options = FirefoxOptions()
                            firefox_options.add_argument("--headless")
                            if firefox_path:
                                firefox_options.binary_location = firefox_path
                            driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=firefox_options)
            
            start_batch_time = datetime.now()
            start_time = datetime.now()
            if THREAD:
                handle_request_pages_threaded(conn, query, NODOC, WORKERS, BATCH_SIZE, STAGING)
            else:
                with requests.Session() as session:
                    handle_request_pages(conn, session, STAGING, query, NODOC, False) #, WORKERS, BATCH_SIZE)

        elif args.search:
            

            page_number = 1
            current_url = f"{base_url}&page={page_number}#search-results"
            safe_print(f"Handling search pages ...")
            max_pages = -1  # Adjust as needed
            inserted_recs = 0
            updated_recs = 0
            title_tags = {}
            start_time = datetime.now()
            tabprefix = "\t"
            sql_results = []
            
            # get all Titles
            sql = """
                SELECT DISTINCT Title, tag_key, tag_value from requests_tags where date(validSince) <= date('now') and date(validUntil) > date('now') group by Title order by tag_key
            """
            with sqlite3.connect(args.db, isolation_level=None, check_same_thread=False) as conn:
                cursor = conn.cursor()
                Titles = cursor.execute(sql).fetchall()  # Use fetchall() to get results
                for Title in Titles:
                    try:
                        title_tags[Title[0]] = fetch_all_tags_for_title('requests_tags', Title[0], VERBOSE_MODE)
                        existing_tags = True
                    except Exception as e:
                        safe_print(f"{tabprefix}- retrieving existing tags for all titles raised an Exception, {e}")
                        existing_tags = False
                        pass
                        
            
            url = current_url
            while True:
                if page_number > max_pages and max_pages > 0:
                    if VERBOSE_MODE:
                        print(f"Reached max pages limit: {max_pages}")
                    break
                
                # safe_print(f"Fetching search page: {current_url.split('&')[-1]} ... ", end='')
                msg = f"{tabprefix}- fetching search page {page_number} ... "
                try:
                    with requests.Session() as session:
                        result = fetch_page(current_url, session)
                        if result:
                            return_status = result[0]
                            soup = result[1]
                            navigation_div = soup.find('div', id='js-search-results')
                        else:
                            navigation_div = None
                                        
                    # soup = fetch_page(url)
                    # get_search_results returns record with Title, DecisionDate etc
                    #  return data, inserted_recs, updated_recs, sql_return
                    results = get_search_results(soup, STAGING, title_tags, base_href, conn, VERBOSE_MODE)
                    
                    if results:
                        page_results = results[0]
                        inserted_recs += results[1]
                        updated_recs += results[2]
                        sql_results.append(results[3])
                        
                    else:
                        # if VERBOSE_MODE:
                        safe_print(f"-{tabprefix}- no results found for page {page}. Break.")
                        break
                    
                    
                    # this is a record
                    # print(f"\npage_results: ", page_results)
                    try:
                        all_results.extend(page_results)
                        # total_result = 0
                        # for result in all_results:
                            # total_result+=len(result)
                        safe_print(f"{msg} total inserted {inserted_recs} / updated {updated_recs}")
                        
                    except sqlite3.Error as e:
                        break
                    except TypeError as e:
                        if VERBOSE_MODE:
                            safe_print(f"{tabprefix}- no results found on page {page}.")
                        # break
                        
                    if navigation_div:
                        try:
                            next_page_link = navigation_div.find('a', rel='next')
                            if next_page_link and next_page_link.get('href'):
                                page_number += 1
                                next_page_href = next_page_link['href']
                                next_url = urljoin(current_url, next_page_href)
                                if VERBOSE_MODE:
                                    safe_print(f"{tabprefix}- next page {page_number} link found as {next_url}")
                                current_url = next_url
                                lastpage = False
                                continue  # Go back to the start of the loop to process the new page
                                
                            else:
                                if VERBOSE_MODE:
                                    safe_print(f"{tabprefix}- no next page link found total pages: {page_number}; assumed to be on the last search result page.")
                                # Reset for the next tab
                                page_number = 1
                                current_url = url  # Reset URL to start from the beginning for the next tab
                                lastpage = True
                                break
                                
                        except Exception as e:
                            if VERBOSE_MODE:
                                safe_print(f"{tabprefix}- no next page link found for search, total pages: {page_number}; assumed to be on the last page\n{tabprefix}- resetting {current_url} to {url}.")
                            # Reset for the next tab
                            page_number = 1
                            current_url = url  # Reset URL to start from the beginning for the next tab
                            lastpage = True
                            break
                        # print(f"\nall_results: ", all_results)
                        # backwards compatibility
                        # update_or_insert_tag(page_results, conn)
                    else:
                        safe_print(f"{tabprefix}- navigation div not found")
                        current_url = url
                        break
                    
                    if lastpage:
                        break

                except requests.RequestException as e:
                    if VERBOSE_MODE:
                        safe_print(f"{tabprefix}- failed with error fetching page {page_number}: {e}")
                
                if not page_results:
                    if VERBOSE_MODE:
                        safe_print(f"{tabprefix}- no results found on this page. Break.")
                    break

        else:
            about_message()
            return
        
        end_time = datetime.now()
        print(f"")
        if all_results:
            total_results = 0
            total_sql_results = 0
            for result in all_results:
                total_results += len(result)
            for result in sql_results:
                total_sql_results += len(result)
            
            save_results(sql_results, conn, VERBOSE_MODE, 1, 1, total_sql_results, total_results, 1, False)

            print(f"A total of {total_results} request results were found, with {inserted_recs} inserts and {updated_recs} updates gathered. Time elapsed {end_time-start_time}.")
            
        else:
            print(f"Execution finished at {datetime.now()}. Time elapsed {end_time-start_time}.")

    conn.close()
    return
    
if __name__ == "__main__":
    try:
        with Manager() as manager:
            tag_cache = manager.dict()
            main()
    finally:
        cleanup()

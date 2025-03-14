# wookey
wookey - the key to documents from FOIA requests

this repository holds python3 scripts with which new and changed requests and consequently documents from open.minvws.nl can be detected and locally saved.

```
wookey.py
Starting script: wookey.py to process Woo documents from open.minvws.nl.
usage: wookey.py [-h] [--about] --db DB [--download [download_path]] [--force] [--files FILES] 
                 [--table [tags]] [--tags [tags]] [--debug] [--verbose] [--silent] [--staging]
                 [--search] [--requests] [--eval] [--period PERIOD] [--nodoc] [--nowritetocache] 
                 [--query [SQL_QUERY]] [--limit [LIMIT_QUERY]] [--thread] [--wait [WAIT_MILLISEC]] 
                 [--batch [BATCH_SIZE]] [--workers [WORKERS]]
```
as you can see there are a lot of options of which a lot might not work. that's how this story goes.

what should work is this:
```
wookey.py --db YourDatabase.db --search
wookey.py --db YourDatabase.db --requests --query "SELECT DISTINCT rt.Title, rt.tag_value FROM requests_tags rt WHERE rt.tag_key = 'DossierURL' ORDER BY rt.ModifiedDate DESC"
wookey.py --db YourDatabase.db --download "YourRootFolderForDownloads" --query "SELECT DISTINCT rt.Title AS title, rt.tag_key AS tag_key, COALESCE(rt2.tag_value, '--') AS download_url, 'archieven_files/' || rt.tag_value AS bestandsnaam FROM requests_tags rt LEFT OUTER JOIN requests_tags rt2 ON rt.Title = rt2.Title AND rt2.tag_key = 'Download document archief_linkhref_1' WHERE rt.tag_key = 'Download document archief_linkname_1'"
```

this will get you the downloaded archives, around 260 or something, you'll never know. maybe 350GB? who can say. maybe less. how much time? well, you could try tinkering with parameters like --thread and --workers, but there is really no guarantee that will work and most probably it might ruin things.

moving on to the next script.

```
wookey-archives.py
Starting script: wookey-archives.py to process Archive files from open.minvws.nl.
usage: wookey-archives.py [-h] --db DB [--empty] [--check] [--query QUERY [QUERY ...]]
                          [--text] [--notmpclean] [--noduplicates] [folder_path]
```

again a lot of options. what should work is this:
```
4.  wookey-archives.py --db openminvws_6.0.0.db "YourRootFolderForDownloads\archieven_files"
```
this will add the contents description of every .zip file to a table called documents_archives. it will result in documents_archives having a collection of documents that are *actually* published, with references to the archive they're coming from.
```
5. wookey-archives.py --db YourDatabase.db --text "YourRootFolderForDownloads\archieven_files"
```
this will create the text and ocrText fields for every document. for that it will unpack a zip, scan it's contents, register everything and clean up again. you will have to have tesseract and plotter installed. follow the errors when using the scripts. or have your local AI write you an installer for your specific situation, whatever you're comfortable with.

the final script for this release.

```
wookey-archives-search.py
Starting script: wookey-archives-search.py to process Archive files from open.minvws.nl.
usage: wookey-archives-search.py [-h] --db DB --table TABLE --fields FIELDS --return_fields RETURN_FIELDS
                                 [--search SEARCH] [--case] [--context CONTEXT] [--all] [--save] [--create]
                                 [--any] [--noduplicates] [--emailaddress] [--debug] [--verbose]
```

it's as if the options list never ends. this should work:
```
6. wookey-archives-search.py --db YourDatabase.db --table documents_archives --fields "text" --return_fields "documentName,{results},fromFilePath" --search "Your Eloquent, Search, String" --save --any
```
this will give you an excel file containing references to all documents that have any of the phrases or words "Your Eloquent, Search, String" in the provided search fields, in this case text, but nut ocrText. a more appealing example would be 'vaccinatieregistratie' or 'CIMS', but that's just me.

----

An update. FOIA'd documents on covid-19 used to be published through the website https://wobcovid19.rijksoverheid.nl. That website is gone and only a subset of documents from that website is accessible through the new open.minvws.nl. This script wookey-extract-docnr.py is used to extract the document numbers from the combined PDF files that used to be used. It checks a folder for these PDF's, gathers meta data on the PDF's and when tesseract-OCR is installed will offer to read he document numbers from the PDF's and save the results to a database table called documents_assembled and documents_assembled_pages.

```
usage: wookey-extract-docnr.py [-h] --folder FOLDER --db DB [--force] [--text] [--quiet] [--verbose] [--ocr]
                               [--workers WORKERS] [--length LENGTH] [--quick] [--dpi DPI]

wookey-extract-docnr.py - PDF Document Number Extractor

options:
  -h, --help         show this help message and exit
  --folder FOLDER    Folder containing PDF files, a single PDF file, or a wildcard pattern (e.g., 'path/*.pdf')
  --db DB            Database file name to store results
  --force            Force reprocessing of all PDFs
  --text             Extract and save full text of pages
  --quiet            Suppress all output messages
  --verbose          Verbose messages
  --ocr              Forces OCR on pages
  --workers WORKERS  Number of worker threads (default: 4)
  --length LENGTH    Exact length of the digit sequence in document numbers to match (e.g., 4 for exactly 4 digits)
  --quick            Quick mode: only check first and last lines, prefer structure matching previous number
  --dpi DPI          DPI mode to use, e.g. 300
```

@leon1969, https://x.com/leon1969

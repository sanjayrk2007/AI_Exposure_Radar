
# AI Exposure Radar — M2 Web Collection

## Overview

This module collects publicly accessible vendor website content for the AI Exposure Radar project.

## Components

- `fetcher.py`: Fetches and caches public web responses.
- `discover.py`: Discovers relevant pages using sitemaps, common paths, and homepage links.
- `parse.py`: Extracts readable text from HTML and PDFs.
- `collector.py`: Coordinates collection and returns documents and source reports.
- `contracts/models.py`: Defines shared data structures.

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Tests

```powershell
python -m pytest tests -v
```

## Collection interface

```python
from m2_web.collector import collect_web

documents, reports = collect_web(vendor, plan)
```

## Responsible collection

Collect public pages only. Respect robots.txt and request limits. Do not bypass access controls or submit forms. Local cache snapshots should not be committed to version control.

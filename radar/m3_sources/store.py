"""Reading and writing ``cache/<vendor_id>/documents.json`` and
``source_reports.json`` (plan 4.2 repo layout).

M2 and M3 both write evidence for the same vendor into the SAME two files, so
M3 only ever *merges*: it replaces entries whose ``source_type`` it owns
(CAREERS / REPOS / FILINGS) and leaves every other entry untouched. MANUAL
documents are only added/removed by the manual-inbox functions. Writes are
atomic (temp file + rename) and serialised per file with a lock, so M1's thread
pool can run vendors in parallel safely.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

from radar.contracts.models import Document, SourceReport
from radar.m3_sources.common import cache_root, validate_vendor_id

_DOC_FIELDS = {f.name for f in fields(Document)}
_REPORT_FIELDS = {f.name for f in fields(SourceReport)}

_locks: dict[str, threading.RLock] = {}
_locks_guard = threading.Lock()


def _lock_for(path: Path) -> threading.RLock:
    with _locks_guard:
        return _locks.setdefault(str(path.resolve()), threading.RLock())


def vendor_dir(vendor_id: str, cache_dir: Optional[Union[str, Path]] = None) -> Path:
    root = Path(cache_dir) if cache_dir else cache_root()
    if not isinstance(vendor_id, str) or not vendor_id or "/" in vendor_id or "\\" in vendor_id or ".." in vendor_id:
        raise ValueError(f"invalid vendor_id {vendor_id!r}")
    return root / vendor_id


def documents_path(vendor_id: str, cache_dir: Optional[Union[str, Path]] = None) -> Path:
    return vendor_dir(vendor_id, cache_dir) / "documents.json"


def reports_path(vendor_id: str, cache_dir: Optional[Union[str, Path]] = None) -> Path:
    return vendor_dir(vendor_id, cache_dir) / "source_reports.json"


# ---- (de)serialisation ---------------------------------------------------- #
def document_from_dict(d: dict[str, Any]) -> Document:
    data = {k: v for k, v in d.items() if k in _DOC_FIELDS}
    data.setdefault("meta", {})
    return Document(**data)


def report_from_dict(d: dict[str, Any]) -> SourceReport:
    return SourceReport(**{k: v for k, v in d.items() if k in _REPORT_FIELDS})


def _read_json(path: Path) -> tuple[list[dict[str, Any]], Optional[str]]:
    """Return ``(items, wrapper_key)``. ``wrapper_key`` is set when the file is a
    dict wrapping the list (e.g. ``{"documents": [...]}``) so it can be preserved."""
    if not path.is_file():
        return [], None
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)], None
    if isinstance(data, dict):
        for key in ("documents", "source_reports", "reports", "items"):
            if isinstance(data.get(key), list):
                return [x for x in data[key] if isinstance(x, dict)], key
    return [], None


def _write_json(path: Path, items: list[dict[str, Any]], wrapper_key: Optional[str]) -> None:
    payload: Any = {wrapper_key: items} if wrapper_key else items
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=False)
        fh.write("\n")
    os.replace(tmp, path)


# ---- reads ----------------------------------------------------------------- #
def read_documents(vendor_id: str, source_types: Optional[Sequence[str]] = None,
                   cache_dir: Optional[Union[str, Path]] = None) -> list[Document]:
    path = documents_path(vendor_id, cache_dir)
    with _lock_for(path):
        items, _ = _read_json(path)
    docs = [document_from_dict(d) for d in items]
    if source_types is not None:
        wanted = set(source_types)
        docs = [d for d in docs if d.source_type in wanted]
    return docs


def read_reports(vendor_id: str, source_types: Optional[Sequence[str]] = None,
                 cache_dir: Optional[Union[str, Path]] = None) -> list[SourceReport]:
    path = reports_path(vendor_id, cache_dir)
    with _lock_for(path):
        items, _ = _read_json(path)
    reps = [report_from_dict(d) for d in items]
    if source_types is not None:
        wanted = set(source_types)
        reps = [r for r in reps if r.source_type in wanted]
    return reps


# ---- writes ---------------------------------------------------------------- #
def write_source_results(vendor_id: str, source_type: str, docs: Iterable[Document],
                         report: SourceReport,
                         cache_dir: Optional[Union[str, Path]] = None) -> None:
    """Replace this vendor's cached documents and report for ``source_type``,
    keeping every other source type's entries as they are."""
    validate_vendor_id(vendor_id)
    docs = list(docs)
    for d in docs:
        if d.source_type != source_type or d.vendor_id != vendor_id:
            raise ValueError(f"document {d.doc_id} does not belong to {vendor_id}/{source_type}")
    dpath, rpath = documents_path(vendor_id, cache_dir), reports_path(vendor_id, cache_dir)
    with _lock_for(dpath):
        items, wrap = _read_json(dpath)
        kept = [x for x in items if x.get("source_type") != source_type]
        merged = kept + [asdict(d) for d in sorted(docs, key=lambda d: d.doc_id)]
        _write_json(dpath, merged, wrap)
    with _lock_for(rpath):
        items, wrap = _read_json(rpath)
        kept = [x for x in items if x.get("source_type") != source_type]
        _write_json(rpath, kept + [asdict(report)], wrap)


def upsert_document(doc: Document, cache_dir: Optional[Union[str, Path]] = None) -> None:
    """Insert or replace one document (matched on ``doc_id``)."""
    validate_vendor_id(doc.vendor_id)
    path = documents_path(doc.vendor_id, cache_dir)
    with _lock_for(path):
        items, wrap = _read_json(path)
        items = [x for x in items if x.get("doc_id") != doc.doc_id]
        items.append(asdict(doc))
        _write_json(path, items, wrap)


def remove_document(vendor_id: str, doc_id: str, source_type: str,
                    cache_dir: Optional[Union[str, Path]] = None) -> bool:
    """Delete one document by id (restricted to ``source_type``). True if removed."""
    validate_vendor_id(vendor_id)
    path = documents_path(vendor_id, cache_dir)
    with _lock_for(path):
        items, wrap = _read_json(path)
        kept = [x for x in items if not (x.get("doc_id") == doc_id and x.get("source_type") == source_type)]
        if len(kept) == len(items):
            return False
        _write_json(path, kept, wrap)
        return True


# ---- M4 compatibility helpers -------------------------------------------- #
def read_all_documents(cache_dir: Optional[Union[str, Path]] = None) -> list[Document]:
    """Read all cached documents across all vendors in cache_dir."""
    root = Path(cache_dir) if cache_dir else cache_root()
    if not root.is_dir():
        return []
    all_docs: list[Document] = []
    for vendor_entry in sorted(root.iterdir()):
        if vendor_entry.is_dir() and (vendor_entry / "documents.json").is_file():
            all_docs.extend(read_documents(vendor_entry.name, cache_dir=root))
    return all_docs


def export_cache_to_manifest(output_path: Union[str, Path],
                             cache_dir: Optional[Union[str, Path]] = None) -> int:
    """Export all cached vendor documents to a JSONL manifest compatible with M4."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    docs = read_all_documents(cache_dir)
    with out.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")
    return len(docs)


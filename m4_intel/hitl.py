"""Backend Human-in-the-Loop (HITL) review store for M4 LLM proposals.

Design contracts
----------------
* Proposals written by the LLM pipeline are always stored with
  ``review_status = "LLM_PROPOSED_UNREVIEWED"``.
* Only an explicit call to ``record_decision`` with a valid decision string
  ("CONFIRM", "OVERRIDE", "REJECT") changes a proposal's status.
* Code never sets ``review_status`` to "HUMAN_REVIEWED_*" automatically.
* Proposals and decisions live in separate JSON Lines files so the original
  proposal record is never mutated after it is written.
* The audit log appends an immutable entry for every decision event.
* No Streamlit, FastAPI, or other UI is included; this module exposes a
  pure-Python API that a separate UI team can call.

File layout inside ``store_dir``
---------------------------------
::

    proposals.jsonl   - one JSON object per line, written by record_proposal()
    decisions.jsonl   - one JSON object per line, written by record_decision()
    audit.jsonl       - one JSON object per line, immutable audit trail
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

VALID_DECISIONS = {"CONFIRM", "OVERRIDE", "REJECT"}
PROPOSAL_STATUS_UNREVIEWED = "LLM_PROPOSED_UNREVIEWED"
HUMAN_REVIEWED_PREFIX = "HUMAN_REVIEWED_"

_OVERRIDE_REASON_CODES = {
    "CURRENT_SPECIFIC_USE", "NAMES_PROVIDER", "VAGUE_MARKETING",
    "FUTURE_OR_HEDGED", "NOT_VENDOR_OWN_OFFERING", "NOT_AI",
}


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append a single JSON record to a JSONL file."""
    line = json.dumps(record, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read all records from a JSONL file; malformed lines are skipped."""
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                records.append(obj)
        except json.JSONDecodeError:
            continue
    return records


def _utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _validate_override(value: dict[str, Any]) -> None:
    """Raise ValueError if the override dict is malformed."""
    if not isinstance(value.get("is_technical_evidence"), bool):
        raise ValueError("override_value must contain 'is_technical_evidence' (bool)")
    reason_code = value.get("reason_code")
    if reason_code not in _OVERRIDE_REASON_CODES:
        raise ValueError(
            f"override_value['reason_code'] must be one of {sorted(_OVERRIDE_REASON_CODES)}"
        )


class HITLStore:
    """Persist LLM proposals and human review decisions separately.

    Parameters
    ----------
    store_dir:
        Directory where proposals.jsonl, decisions.jsonl, and audit.jsonl
        are kept. Created on first write.
    """

    def __init__(self, store_dir: str | Path) -> None:
        self.store_dir = Path(store_dir)
        self._proposals = self.store_dir / "proposals.jsonl"
        self._decisions = self.store_dir / "decisions.jsonl"
        self._audit = self.store_dir / "audit.jsonl"

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def record_proposal(
        self,
        signal_id: str,
        vendor_id: str,
        excerpt: str,
        proposal: dict[str, Any],
    ) -> str:
        """Persist one LLM proposal. Returns the generated proposal_id.

        The stored record always has review_status = PROPOSAL_STATUS_UNREVIEWED;
        any review_status value supplied inside proposal is overwritten.
        """
        if not isinstance(signal_id, str) or not signal_id.strip():
            raise ValueError("signal_id must be a non-empty string")
        if not isinstance(vendor_id, str) or not vendor_id.strip():
            raise ValueError("vendor_id must be a non-empty string")
        if not isinstance(excerpt, str) or not excerpt.strip():
            raise ValueError("excerpt must be a non-empty string")
        if not isinstance(proposal, dict):
            raise TypeError("proposal must be a dict")

        proposal_id = str(uuid.uuid4())
        record: dict[str, Any] = {
            "proposal_id": proposal_id,
            "signal_id": signal_id,
            "vendor_id": vendor_id,
            "excerpt": excerpt,
            "proposal": {**proposal, "review_status": PROPOSAL_STATUS_UNREVIEWED},
            "created_at": _utcnow(),
        }
        _append_jsonl(self._proposals, record)
        return proposal_id

    def record_decision(
        self,
        proposal_id: str,
        decision: str,
        rationale: str,
        reviewer_id: str,
        override_value: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record an explicit human review decision for a proposal.

        Parameters
        ----------
        proposal_id:
            The ID returned by record_proposal().
        decision:
            One of "CONFIRM", "OVERRIDE", "REJECT".
        rationale:
            Free-text explanation; must not be empty.
        reviewer_id:
            Opaque identifier for the reviewing analyst.
        override_value:
            Required when decision == "OVERRIDE". Must be a dict with at
            least is_technical_evidence (bool) and reason_code (str).

        Returns the complete decision record written to disk.
        """
        decision = decision.strip().upper()
        if decision not in VALID_DECISIONS:
            raise ValueError(
                f"decision must be one of {sorted(VALID_DECISIONS)}, got {decision!r}"
            )
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("rationale must be a non-empty string")
        if not isinstance(reviewer_id, str) or not reviewer_id.strip():
            raise ValueError("reviewer_id must be a non-empty string")
        if decision == "OVERRIDE":
            if not isinstance(override_value, dict):
                raise ValueError(
                    "override_value (dict) is required when decision is OVERRIDE"
                )
            _validate_override(override_value)

        proposals = self.get_proposals()
        matched = next(
            (p for p in proposals if p.get("proposal_id") == proposal_id), None
        )
        if matched is None:
            raise KeyError(f"No proposal found with proposal_id={proposal_id!r}")

        review_status = HUMAN_REVIEWED_PREFIX + decision
        decision_record: dict[str, Any] = {
            "decision_id": str(uuid.uuid4()),
            "proposal_id": proposal_id,
            "signal_id": matched.get("signal_id"),
            "vendor_id": matched.get("vendor_id"),
            "decision": decision,
            "rationale": rationale.strip(),
            "reviewer_id": reviewer_id.strip(),
            "review_status": review_status,
            "decided_at": _utcnow(),
        }
        if override_value is not None:
            decision_record["override_value"] = override_value

        _append_jsonl(self._decisions, decision_record)

        audit_entry: dict[str, Any] = {
            "event": "HUMAN_DECISION",
            "decision_id": decision_record["decision_id"],
            "proposal_id": proposal_id,
            "signal_id": matched.get("signal_id"),
            "vendor_id": matched.get("vendor_id"),
            "decision": decision,
            "reviewer_id": reviewer_id.strip(),
            "timestamp": _utcnow(),
        }
        _append_jsonl(self._audit, audit_entry)
        return decision_record

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def get_proposals(self) -> list[dict[str, Any]]:
        """Return all stored proposals (unreviewed and reviewed)."""
        return _read_jsonl(self._proposals)

    def get_decisions(self) -> list[dict[str, Any]]:
        """Return all stored decisions in insertion order."""
        return _read_jsonl(self._decisions)

    def get_audit(self) -> list[dict[str, Any]]:
        """Return the full, immutable audit log."""
        return _read_jsonl(self._audit)

    def get_pending_proposals(self) -> list[dict[str, Any]]:
        """Return proposals that have no corresponding decision record yet."""
        decided_ids = {d.get("proposal_id") for d in self.get_decisions()}
        return [p for p in self.get_proposals() if p.get("proposal_id") not in decided_ids]

    def get_decision_for_proposal(self, proposal_id: str) -> dict[str, Any] | None:
        """Return the most recent decision for proposal_id, or None."""
        result: dict[str, Any] | None = None
        for decision in self.get_decisions():
            if decision.get("proposal_id") == proposal_id:
                result = decision
        return result

    def reviewed_signal_ids(self) -> set[str]:
        """Return signal_ids that have at least one human decision recorded."""
        return {d.get("signal_id") for d in self.get_decisions() if d.get("signal_id")}

from __future__ import annotations

from collections import Counter
from html import escape
from typing import Any, Iterable

import streamlit as st

from radar.m1_app.theme import (
    BRICK,
    MOSS,
    OCHRE,
    RISK_COLORS,
    STONE,
    STRENGTH_COLORS,
    TIER_COLORS,
    USAGE_COLORS,
)

RISK_ORDER = ["Critical", "High", "Medium", "Low", "None identified", "Not assessed"]
TIER_ORDER = ["Critical", "High", "Medium", "Low"]
USAGE_ORDER = ["Yes", "Inconclusive", "No"]
COVERAGE_TARGET = 70.0
PRIORITY_LIMIT = 5


def record_from_result(vendor: Any, result: Any) -> dict[str, Any]:
    verdict = result.verdict
    risk = result.risk
    return {
        "vendor_id": vendor.vendor_id,
        "name": vendor.name,
        "criticality": result.criticality.tier,
        "ai_usage": verdict.ai_usage,
        "strength": verdict.strength,
        "coverage": round(verdict.coverage * 100, 1),
        "risk": risk.risk_class,
        "documents": len(result.documents),
        "signals": len(result.signals),
        "providers": list(verdict.providers),
        "use_cases": list(verdict.use_cases),
        "reasons": list(verdict.reasons),
        "rationale": risk.rationale,
        "action": risk.recommended_action,
        "error": "",
    }


def record_from_error(vendor: Any, message: str) -> dict[str, Any]:
    return {
        "vendor_id": vendor.vendor_id,
        "name": vendor.name,
        "criticality": "Not assessed",
        "ai_usage": "Inconclusive",
        "strength": "None",
        "coverage": 0.0,
        "risk": "Not assessed",
        "documents": 0,
        "signals": 0,
        "providers": [],
        "use_cases": [],
        "reasons": ["Pipeline failed; no conclusion can be drawn."],
        "rationale": "",
        "action": "",
        "error": message,
    }


def _rank(order: list[str], value: str) -> int:
    return order.index(value) if value in order else len(order)


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    return f"{count} {singular if count == 1 else plural or singular + 's'}"


def summarize(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(records)
    done = [r for r in records if not r.get("error")]
    usage = Counter(r["ai_usage"] for r in done)
    risk = Counter(r["risk"] for r in records)
    tier = Counter(r["criticality"] for r in records)
    providers = Counter(p for r in done for p in r.get("providers", []))
    matrix: dict[str, dict[str, list[str]]] = {
        t: {u: [] for u in USAGE_ORDER} for t in TIER_ORDER
    }
    for r in done:
        if r["criticality"] in matrix and r["ai_usage"] in matrix[r["criticality"]]:
            matrix[r["criticality"]][r["ai_usage"]].append(r["vendor_id"])
    coverage = [r["coverage"] for r in done]
    ordered = sorted(
        records,
        key=lambda r: (
            bool(r.get("error")),
            _rank(RISK_ORDER, r["risk"]),
            _rank(TIER_ORDER, r["criticality"]),
            r["coverage"],
        ),
    )
    return {
        "total": len(records),
        "completed": len(done),
        "errors": len(records) - len(done),
        "usage": usage,
        "risk": risk,
        "tier": tier,
        "ai_yes": usage.get("Yes", 0),
        "high_risk": risk.get("Critical", 0) + risk.get("High", 0),
        "avg_coverage": round(sum(coverage) / len(coverage), 1) if coverage else 0.0,
        "low_coverage": sum(1 for c in coverage if c < COVERAGE_TARGET),
        "providers": providers.most_common(6),
        "matrix": matrix,
        "ordered": ordered,
        "priority": [r for r in ordered if not r.get("error")][:PRIORITY_LIMIT],
    }


def headline(summary: dict[str, Any]) -> str:
    if not summary["completed"]:
        return "No vendors could be assessed in this run."
    text = (
        f"{summary['ai_yes']} of {_plural(summary['completed'], 'assessed vendor')} "
        f"show evidence of AI use, and {summary['high_risk']} "
        f"{'is' if summary['high_risk'] == 1 else 'are'} rated High or Critical risk."
    )
    if summary["low_coverage"]:
        text += (
            f" {_plural(summary['low_coverage'], 'vendor')} fell below the "
            f"{COVERAGE_TARGET:.0f}% coverage threshold, so absence of evidence there is not a clean result."
        )
    if summary["errors"]:
        text += f" {_plural(summary['errors'], 'vendor')} failed to process."
    return text


def _pill(label: str, colors: dict[str, str]) -> str:
    return f'<span class="pill" style="--c:{colors.get(label, STONE)}">{escape(label)}</span>'


def _meter(value: float) -> str:
    color = MOSS if value >= COVERAGE_TARGET else OCHRE if value >= 40 else BRICK
    return (
        '<div class="meter"><div class="meter-track">'
        f'<div class="meter-fill" style="width:{min(value, 100):.0f}%;background:{color}"></div>'
        f'<div class="meter-tick" style="left:{COVERAGE_TARGET:.0f}%"></div></div>'
        f"<span>{value:.0f}%</span></div>"
    )


def _kpi(label: str, value: str, note: str, color: str) -> str:
    return (
        f'<div class="kpi" style="--c:{color}"><div class="kpi-label">{escape(label)}</div>'
        f'<div class="kpi-value">{escape(value)}</div><div class="kpi-note">{escape(note)}</div></div>'
    )


def _kpis(s: dict[str, Any]) -> str:
    inconclusive = s["usage"].get("Inconclusive", 0)
    no = s["usage"].get("No", 0)
    failed = f" · {s['errors']} failed" if s["errors"] else ""
    cards = [
        _kpi("Vendors assessed", str(s["completed"]), f"of {s['total']} in inventory{failed}", "#1E2321"),
        _kpi("AI use evidenced", str(s["ai_yes"]), f"{inconclusive} inconclusive · {no} no", "#2E5E4E"),
        _kpi("High / Critical risk", str(s["high_risk"]), "for priority analyst review", BRICK),
        _kpi("Avg. source coverage", f"{s['avg_coverage']:.0f}%", f"{s['low_coverage']} below {COVERAGE_TARGET:.0f}% threshold", OCHRE),
    ]
    return '<div class="grid g4">' + "".join(cards) + "</div>"


def _stack(title: str, counts: Counter, order: list[str], colors: dict[str, str]) -> str:
    present = [k for k in order if counts.get(k)]
    if not present:
        body = '<div class="empty">No data.</div>'
    else:
        segments = "".join(
            f'<div class="seg" style="flex:{counts[k]};background:{colors.get(k, STONE)}" '
            f'title="{escape(k)}: {counts[k]}"></div>'
            for k in present
        )
        legend = "".join(
            f'<li><i style="background:{colors.get(k, STONE)}"></i>{escape(k)}<b>{counts[k]}</b></li>'
            for k in present
        )
        body = f'<div class="stack">{segments}</div><ul class="legend">{legend}</ul>'
    return f'<div class="card"><div class="card-title">{escape(title)}</div>{body}</div>'


def _matrix(matrix: dict[str, dict[str, list[str]]]) -> str:
    peak = max((len(ids) for row in matrix.values() for ids in row.values()), default=0) or 1
    head = "<th></th>" + "".join(f"<th>{u}</th>" for u in USAGE_ORDER)
    rows = ""
    for tier in TIER_ORDER:
        cells = f'<td class="rowh">{_pill(tier, TIER_COLORS)}</td>'
        for usage in USAGE_ORDER:
            ids = matrix[tier][usage]
            if ids:
                alpha = 0.12 + 0.58 * len(ids) / peak
                cells += (
                    f'<td style="background:rgba(180,83,26,{alpha:.2f})">'
                    f'<div class="cell-n">{len(ids)}</div>'
                    f'<div class="cell-ids">{escape(", ".join(ids))}</div></td>'
                )
            else:
                cells += '<td style="background:#F3EEE0"><span class="cell-empty">–</span></td>'
        rows += f"<tr>{cells}</tr>"
    return (
        '<div class="card"><div class="card-title">Criticality × AI usage</div>'
        f'<table class="matrix"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'
    )


def _providers(items: list[tuple[str, int]]) -> str:
    if not items:
        body = '<div class="empty">No AI providers were named in verified evidence.</div>'
    else:
        peak = items[0][1]
        body = '<div class="bars">' + "".join(
            f'<div class="bar-row"><span>{escape(name)}</span>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{100 * count / peak:.0f}%"></div></div>'
            f'<span class="bar-n">{count}</span></div>'
            for name, count in items
        ) + "</div>"
    return f'<div class="card"><div class="card-title">Named AI providers (vendors)</div>{body}</div>'


def _priority(records: list[dict[str, Any]]) -> str:
    if not records:
        return '<div class="empty">Nothing to prioritise.</div>'
    out = ""
    for r in records:
        action = r["action"] or "No recommended action was produced for this vendor."
        out += (
            f'<div class="prio" style="--c:{RISK_COLORS.get(r["risk"], STONE)}">'
            f'<div class="prio-head"><span class="prio-id">{escape(r["vendor_id"])}</span>'
            f'<strong>{escape(r["name"])}</strong>'
            f'{_pill(r["risk"] + " risk", {r["risk"] + " risk": RISK_COLORS.get(r["risk"], STONE)})}'
            f'{_pill(r["criticality"], TIER_COLORS)}{_pill(r["ai_usage"], USAGE_COLORS)}</div>'
            f"<p>{escape(action)}</p></div>"
        )
    return out


def _table(records: list[dict[str, Any]]) -> str:
    head = "".join(
        f"<th>{h}</th>"
        for h in ["Vendor", "Criticality", "AI usage", "Strength", "Risk", "Coverage", "Docs", "Signals"]
    )
    rows = ""
    for r in records:
        if r.get("error"):
            risk = f'<span class="pill" style="--c:{BRICK}" title="{escape(r["error"])}">Failed</span>'
        else:
            risk = _pill(r["risk"], RISK_COLORS)
        rows += (
            f'<tr><td><span class="vname">{escape(r["name"])}</span>'
            f'<span class="vid">{escape(r["vendor_id"])}</span></td>'
            f'<td>{_pill(r["criticality"], TIER_COLORS)}</td>'
            f'<td>{_pill(r["ai_usage"], USAGE_COLORS)}</td>'
            f'<td>{_pill(r["strength"], STRENGTH_COLORS)}</td>'
            f"<td>{risk}</td><td>{_meter(r['coverage'])}</td>"
            f'<td class="num">{r["documents"]}</td><td class="num">{r["signals"]}</td></tr>'
        )
    return f'<div class="tbl-wrap"><table class="tbl"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'


def _detail(r: dict[str, Any]) -> str:
    def block(label: str, body: str) -> str:
        return f'<div class="detail-label">{label}</div>{body}'

    parts = [
        f'<div>{_pill(r["risk"], RISK_COLORS)} {_pill(r["criticality"], TIER_COLORS)} '
        f'{_pill(r["ai_usage"], USAGE_COLORS)} {_pill(r["strength"], STRENGTH_COLORS)}</div>'
    ]
    if r.get("error"):
        parts.append(block("Run error", f'<div class="detail-text">{escape(r["error"])}</div>'))
    if r["rationale"]:
        parts.append(block("Risk rationale", f'<div class="detail-text">{escape(r["rationale"])}</div>'))
    if r["action"]:
        parts.append(block("Recommended action", f'<div class="detail-text">{escape(r["action"])}</div>'))
    if r["providers"]:
        chips = "".join(f'<span class="chip">{escape(p)}</span>' for p in r["providers"])
        parts.append(block("AI providers", f'<div class="chips">{chips}</div>'))
    if r["use_cases"]:
        items = "".join(f"<li>{escape(u)}</li>" for u in r["use_cases"])
        parts.append(block("Observed use cases", f'<ul class="detail-list">{items}</ul>'))
    if r["reasons"]:
        items = "".join(f"<li>{escape(x)}</li>" for x in r["reasons"])
        parts.append(block("Verdict reasoning", f'<ul class="detail-list">{items}</ul>'))
    return "".join(parts)


def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def render_dashboard(records: list[dict[str, Any]]) -> None:
    s = summarize(records)
    _html(f'<div class="headline">{escape(headline(s))}</div>')
    overview, vendors, detail = st.tabs(["Overview", "Vendors", "Vendor detail"])

    with overview:
        _html(_kpis(s))
        _html(
            '<div class="grid g2">'
            + _stack("Risk distribution", s["risk"], RISK_ORDER, RISK_COLORS)
            + _stack("AI usage verdict", s["usage"], USAGE_ORDER, USAGE_COLORS)
            + "</div>"
        )
        _html('<div class="grid g2">' + _matrix(s["matrix"]) + _providers(s["providers"]) + "</div>")
        _html('<div class="section">Priority review queue</div>' + _priority(s["priority"]))

    with vendors:
        _html(_table(s["ordered"]))

    with detail:
        for r in s["ordered"]:
            label = f"{r['vendor_id']} · {r['name']} — {'Failed' if r.get('error') else r['risk']}"
            with st.expander(label):
                _html(_detail(r))

"""Lightweight fallback bridge for YAML operations."""
import io
from ruamel.yaml import YAML

_y = YAML(typ="safe")
safe_load = _y.load
load = _y.load

def dump(data, stream=None, **kwargs):
    y = YAML()
    if stream is None:
        buf = io.StringIO()
        y.dump(data, buf)
        return buf.getvalue()
    y.dump(data, stream)

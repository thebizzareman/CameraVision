"""
security.py — Enterprise security primitives for CameraVision.

Provides:
  - Session token generation and verification
  - Rate limiting (token-bucket per client)
  - HTML escaping for dashboard output
  - JSON Schema validation for gesture_db.json
  - Input sanitization and URL allow-listing

No external dependencies beyond the Python standard library.
"""

import html
import json
import os
import secrets
import threading
import time
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Session authentication
# ---------------------------------------------------------------------------

SESSION_TOKEN_BYTES = 32  # 256-bit entropy


def generate_token():
    """Return a URL-safe, 256-bit random session token."""
    return secrets.token_urlsafe(SESSION_TOKEN_BYTES)


class TokenStore:
    """Thread-safe in-memory session token store."""

    def __init__(self, max_sessions=16, ttl_seconds=3600):
        self._tokens = {}          # token -> creation_time
        self._lock = threading.Lock()
        self._max = max_sessions
        self._ttl = ttl_seconds

    def create(self):
        """Create and return a new session token."""
        tok = generate_token()
        with self._lock:
            # Prune old/expired if at capacity
            now = time.time()
            expired = [t for t, c in self._tokens.items() if now - c > self._ttl]
            for t in expired:
                del self._tokens[t]
            if len(self._tokens) >= self._max:
                # Evict oldest
                oldest = min(self._tokens, key=self._tokens.get)
                del self._tokens[oldest]
            self._tokens[tok] = now
        return tok

    def valid(self, tok):
        """Return True if token is present and not expired."""
        if not tok or not isinstance(tok, str) or len(tok) > 128:
            return False
        with self._lock:
            created = self._tokens.get(tok)
            if created is None:
                return False
            if time.time() - created > self._ttl:
                del self._tokens[tok]
                return False
            return True

    def revoke(self, tok):
        with self._lock:
            self._tokens.pop(tok, None)


# ---------------------------------------------------------------------------
# Rate limiting — token-bucket per client IP
# ---------------------------------------------------------------------------

class RateLimiter:
    """Token-bucket rate limiter.  burst = max requests in a burst window;
    refill = tokens per second."""

    def __init__(self, burst=10, refill=2.0, window_seconds=60):
        self._burst = burst
        self._refill = refill
        self._window = window_seconds
        self._buckets = {}  # ip -> (tokens, last_update)
        self._lock = threading.Lock()

    def _tokens_for(self, ip):
        now = time.time()
        tokens, last = self._buckets.get(ip, (self._burst, now))
        delta = now - last
        tokens = min(self._burst, tokens + delta * self._refill)
        return tokens, now

    def allow(self, ip):
        """Return True if request from ip is allowed."""
        if not ip:
            return False
        with self._lock:
            tokens, now = self._tokens_for(ip)
            if tokens >= 1.0:
                self._buckets[ip] = (tokens - 1.0, now)
                return True
            self._buckets[ip] = (tokens, now)
            return False

    def cleanup(self, max_age=300):
        """Remove stale entries older than max_age seconds."""
        now = time.time()
        with self._lock:
            stale = [ip for ip, (_, last) in self._buckets.items()
                     if now - last > max_age]
            for ip in stale:
                del self._buckets[ip]


# ---------------------------------------------------------------------------
# Output encoding / XSS prevention
# ---------------------------------------------------------------------------

def escape_html(text):
    """Escape a string for safe insertion into HTML text or attribute context."""
    if text is None:
        return ""
    return html.escape(str(text), quote=True)


def escape_js_string(text):
    """Escape a string for safe insertion into a JavaScript string literal."""
    if text is None:
        return ""
    # JSON string encoding is the safest way to escape for JS
    return json.dumps(str(text))[1:-1]  # strip outer quotes


# ---------------------------------------------------------------------------
# JSON Schema validation for gesture_db.json
# ---------------------------------------------------------------------------

DB_SCHEMA = {
    "type": "object",
    "required": ["version", "entries"],
    "properties": {
        "version": {"type": "integer", "minimum": 1, "maximum": 1000},
        "updated": {"type": "string", "maxLength": 32},
        "comment": {"type": "string", "maxLength": 5000},
        "entries": {
            "type": "array",
            "maxItems": 10000,
            "items": {
                "type": "object",
                "required": ["id", "name", "type", "status", "detector"],
                "properties": {
                    "id": {"type": "string", "maxLength": 128, "pattern": "^[a-z0-9-]+$"},
                    "name": {"type": "string", "minLength": 1, "maxLength": 256},
                    "display_name": {"type": "string", "maxLength": 256},
                    "type": {"type": "string", "maxLength": 64},
                    "origin": {"type": "string", "maxLength": 64},
                    "language": {"type": "string", "maxLength": 16},
                    "culture": {"type": "string", "maxLength": 256},
                    "description": {"type": "string", "maxLength": 2000},
                    "display": {"type": "string", "maxLength": 32},
                    "status": {"type": "string", "maxLength": 16},
                    "synonyms": {
                        "type": "array",
                        "maxItems": 100,
                        "items": {"type": "string", "maxLength": 256}
                    },
                    "sources": {
                        "type": "array",
                        "maxItems": 100,
                        "items": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string", "maxLength": 2048},
                                "ref": {"type": "string", "maxLength": 512},
                                "retrieved": {"type": "string", "maxLength": 32},
                                "note": {"type": "string", "maxLength": 1000}
                            }
                        }
                    },
                    "detector": {"type": "object"},
                    "notes": {"type": "string", "maxLength": 2000}
                },
                "additionalProperties": False
            }
        }
    },
    "additionalProperties": False
}


def _validate_type(val, spec, path):
    """Simple JSON Schema subset validator. Returns list of error strings."""
    errors = []
    if val is None and spec.get("type") != "null":
        # null is only valid if explicitly allowed
        pass

    t = spec.get("type")
    if t == "object" and not isinstance(val, dict):
        errors.append(f"{path}: expected object, got {type(val).__name__}")
    elif t == "array" and not isinstance(val, list):
        errors.append(f"{path}: expected array, got {type(val).__name__}")
    elif t == "string" and not isinstance(val, str):
        errors.append(f"{path}: expected string, got {type(val).__name__}")
    elif t == "integer" and not isinstance(val, int):
        errors.append(f"{path}: expected integer, got {type(val).__name__}")
    elif t == "number" and not isinstance(val, (int, float)):
        errors.append(f"{path}: expected number, got {type(val).__name__}")
    elif t == "boolean" and not isinstance(val, bool):
        errors.append(f"{path}: expected boolean, got {type(val).__name__}")

    if isinstance(val, str):
        if "maxLength" in spec and len(val) > spec["maxLength"]:
            errors.append(f"{path}: string length {len(val)} exceeds max {spec['maxLength']}")
        if "minLength" in spec and len(val) < spec["minLength"]:
            errors.append(f"{path}: string length {len(val)} below min {spec['minLength']}")
        if "pattern" in spec:
            import re
            if not re.match(spec["pattern"], val):
                errors.append(f"{path}: string '{val}' does not match pattern {spec['pattern']}")
    elif isinstance(val, (int, float)):
        if "minimum" in spec and val < spec["minimum"]:
            errors.append(f"{path}: value {val} below minimum {spec['minimum']}")
        if "maximum" in spec and val > spec["maximum"]:
            errors.append(f"{path}: value {val} above maximum {spec['maximum']}")
    elif isinstance(val, list):
        if "maxItems" in spec and len(val) > spec["maxItems"]:
            errors.append(f"{path}: array length {len(val)} exceeds max {spec['maxItems']}")
        if "items" in spec:
            for i, item in enumerate(val):
                errors.extend(_validate_type(item, spec["items"], f"{path}[{i}]"))
    elif isinstance(val, dict):
        if "required" in spec:
            for key in spec["required"]:
                if key not in val:
                    errors.append(f"{path}: missing required key '{key}'")
        if "properties" in spec:
            for key, child_spec in spec["properties"].items():
                if key in val:
                    errors.extend(_validate_type(val[key], child_spec, f"{path}.{key}"))
        if spec.get("additionalProperties") is False:
            allowed = set(spec.get("properties", {}).keys())
            for key in val:
                if key not in allowed:
                    errors.append(f"{path}: unknown key '{key}'")
    return errors


def validate_gesture_db(data):
    """Validate a loaded gesture_db dict against the schema.
    Returns (ok: bool, errors: list of str)."""
    errors = _validate_type(data, DB_SCHEMA, "root")
    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# URL allow-listing for research pipeline
# ---------------------------------------------------------------------------

ALLOWED_RESEARCH_HOSTS = {
    "en.wikipedia.org",
    "tr.wikipedia.org",
    "export.arxiv.org",
}


def is_allowed_research_url(url):
    """Return True if url is on the research allow-list."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        if not host:
            return False
        # Allow subdomains of wikipedia
        if host.endswith(".wikipedia.org"):
            return True
        return host in ALLOWED_RESEARCH_HOSTS
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Zip import safety
# ---------------------------------------------------------------------------

MAX_IMPORT_SIZE = 10 * 1024 * 1024      # 10 MB total
MAX_IMPORT_ENTRIES = 1000
MAX_IMPORT_RATIO = 100                   # compression ratio bomb guard


def validate_zip_safety(zf):
    """Validate a zipfile.ZipFile for safe extraction.
    Returns (ok: bool, msg: str)."""
    total_size = 0
    for info in zf.infolist():
        # Path traversal guard
        if info.filename.startswith(("/", "\\")) or ".." in info.filename:
            return False, f"path traversal detected: {info.filename}"
        total_size += info.file_size
        if info.file_size > 0 and info.compress_size > 0:
            ratio = info.file_size / info.compress_size
            if ratio > MAX_IMPORT_RATIO:
                return False, f"compression ratio bomb: {info.filename}"
    if total_size > MAX_IMPORT_SIZE:
        return False, f"total uncompressed size {total_size} exceeds {MAX_IMPORT_SIZE}"
    if len(zf.infolist()) > MAX_IMPORT_ENTRIES:
        return False, f"entry count {len(zf.infolist())} exceeds {MAX_IMPORT_ENTRIES}"
    return True, "ok"


# ---------------------------------------------------------------------------
# CSP header value
# ---------------------------------------------------------------------------

CSP_HEADER = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"

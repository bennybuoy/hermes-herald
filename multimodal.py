"""Multimodal image input validation and normalization for llm_call / llm_direct.

Converts caller-supplied messages (plain string or OpenAI-style content parts)
and optional top-level ``images`` into canonical OpenAI Chat Completions
messages suitable for both Hermes's public ``ctx.llm.complete()`` facade and
``llm_direct`` endpoints.

Security posture (fail-closed, stdlib-only, zero Hermes core imports):
  - Text-only calls remain plain strings for full backward compatibility.
  - Images are allowed only on ``role="user"`` messages.
  - Inline ``data:`` URLs and local image files are restricted to PNG, JPEG,
    GIF, and WebP, verified via magic bytes (never extension alone), and
    capped before base64 decoding or full file reads.
  - Local file paths reject NUL/control characters, ``..`` traversal segments,
    credential filenames/directories, non-regular files (FIFOs, devices,
    directories), and oversized files via ``O_NONBLOCK`` + ``fstat`` + bounded
    chunked reads.
  - Remote image URLs must be ``https://`` without embedded credentials or
    private/loopback/link-local/CGNAT/metadata hosts or IPs.
  - Error messages and helpers never leak raw base64 payloads or credentials.
"""

from __future__ import annotations

import base64
import binascii
import ipaddress
import os
import re
import socket
import stat
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit

SUPPORTED_IMAGE_MIME_TYPES = frozenset({
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
})
ALLOWED_IMAGE_DETAILS = frozenset({"auto", "low", "high"})

MAX_IMAGE_BYTES = 10 * 1024 * 1024        # 10 MiB raw per image
MAX_TOTAL_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MiB raw total per call
MAX_IMAGES_PER_CALL = 10
MAX_IMAGE_URL_LENGTH = 4096
MAX_CONTENT_PARTS_PER_MESSAGE = 32

_READ_CHUNK_BYTES = 256 * 1024

# Explicit URL scheme (2+ chars before ':') vs single-letter Windows drive path.
_URL_SCHEME_RE = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]+):")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[/\\]")

# Redact any base64 data:image/... payload from error text / provider echoes.
# Deliberately excludes whitespace so trailing diagnostics and subsequent
# data: URLs in the same message are preserved and redacted independently.
_DATA_URL_REDACT_RE = re.compile(
    r"(data:image/[A-Za-z0-9.+-]+(?:;[A-Za-z0-9_.+-]+=[A-Za-z0-9_.+-]+)*;base64,)"
    r"[A-Za-z0-9+/=]+",
    re.IGNORECASE,
)

# Unicode full-stop equivalents mapped to '.' by UTS#46 / IDNA / WHATWG URL parsers.
_UNICODE_DOTS_TABLE = str.maketrans({
    "\u3002": ".",  # ideographic full stop
    "\uff0e": ".",  # fullwidth full stop
    "\uff61": ".",  # halfwidth ideographic full stop
})

# Sensitive credential basenames, suffixes, and parent directories refused on local reads.
_SENSITIVE_FILENAMES = frozenset({
    ".env",
    "auth.json",
    "credentials",
    "credentials.json",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "id_rsa",
    "id_rsa.pub",
    "id_ed25519",
    "id_ed25519.pub",
    "id_ecdsa",
    "id_ecdsa.pub",
    "id_dsa",
    "id_dsa.pub",
    "authorized_keys",
    "known_hosts",
    "passwd",
    "shadow",
    "master.key",
})
_SENSITIVE_SUFFIXES = frozenset({
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".jks",
    ".kdbx",
    ".env",
})
_SENSITIVE_DIR_PARTS = frozenset({
    ".ssh",
    ".aws",
    ".gnupg",
    ".kube",
    ".docker",
})

# Hostnames and suffixes blocked for remote https:// image URLs (SSRF defense).
_BLOCKED_HOSTNAMES = frozenset({
    "localhost",
    "metadata",
    "metadata.google.internal",
    "metadata.goog",
    "instance-data",
})
_BLOCKED_HOST_SUFFIXES = (
    ".localhost",
    ".local",
    ".internal",
    ".localdomain",
    ".home.arpa",
)

_IPV4_TRANSLATED_NETWORK = ipaddress.ip_network("::ffff:0:0:0/96")

_BLOCKED_IPV4_NETWORKS = tuple(
    ipaddress.ip_network(cidr)
    for cidr in (
        "0.0.0.0/8",
        "10.0.0.0/8",
        "100.64.0.0/10",    # CGNAT / Tailscale / Alibaba IMDS (100.100.100.200)
        "127.0.0.0/8",
        "169.254.0.0/16",   # Link-local / AWS/GCP/Azure IMDS
        "172.16.0.0/12",
        "192.0.0.0/24",
        "192.0.2.0/24",
        "192.168.0.0/16",
        "198.18.0.0/15",
        "198.51.100.0/24",
        "203.0.113.0/24",
        "224.0.0.0/4",
        "240.0.0.0/4",
    )
)

_BLOCKED_IPV6_NETWORKS = tuple(
    ipaddress.ip_network(cidr)
    for cidr in (
        "::/128",
        "::1/128",
        "::ffff:0:0/96",
        "::ffff:0:0:0/96",
        "64:ff9b::/96",
        "64:ff9b:1::/48",
        "100::/64",
        "2001::/32",
        "2001:2::/48",
        "2001:10::/28",
        "2001:20::/28",
        "2001:db8::/32",
        "2002::/16",
        "fc00::/7",         # Unique local (includes fd00:ec2::254 AWS IPv6 IMDS)
        "fe80::/10",
        "fec0::/10",
        "ff00::/8",
    )
)


# ---------------------------------------------------------------------------
# Reusable JSON Schema fragments for llm_call / llm_direct
# ---------------------------------------------------------------------------

LLM_MESSAGE_CONTENT_SCHEMA: Dict[str, Any] = {
    "anyOf": [
        {"type": "string"},
        {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_CONTENT_PARTS_PER_MESSAGE,
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["text", "image_url"],
                        "description": "Content part type ('text' or 'image_url').",
                    },
                    "text": {
                        "type": "string",
                        "description": "Text body when type='text'.",
                    },
                    "image_url": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": (
                                    "Local image file path (PNG/JPEG/GIF/WebP), "
                                    "base64 data:image/... URL, or public https:// URL."
                                ),
                            },
                            "detail": {
                                "type": "string",
                                "enum": ["auto", "low", "high"],
                                "description": "Optional vision detail hint.",
                            },
                        },
                        "required": ["url"],
                    },
                },
                "required": ["type"],
            },
        },
    ],
    "description": (
        "Message content: plain text string, or (for role='user') an array of "
        "OpenAI-style content parts ({type:'text', text} and "
        "{type:'image_url', image_url:{url, detail?}}). Local image paths are "
        "validated and base64-encoded automatically (PNG/JPEG/GIF/WebP; max "
        "10 MiB each, 20 MiB total, up to 10 images per call)."
    ),
}

LLM_IMAGES_PARAM_SCHEMA: Dict[str, Any] = {
    "type": "array",
    "maxItems": MAX_IMAGES_PER_CALL,
    "items": {
        "type": "string",
        "description": (
            "Local image file path (PNG/JPEG/GIF/WebP), base64 "
            "data:image/... URL, or public https:// image URL."
        ),
    },
    "description": (
        "Optional images attached to the last user message (up to 10 images; "
        "max 10 MiB each, 20 MiB total raw bytes). Accepts local file paths "
        "(read and base64-encoded automatically), data:image/...;base64,... "
        "URLs, or public https:// URLs. Lets simple callers keep "
        "messages[].content as plain strings."
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def redact_data_urls(text: str) -> str:
    """Replace inline base64 image data payloads with a short placeholder."""
    if not text:
        return ""
    return _DATA_URL_REDACT_RE.sub(r"\1[redacted]", str(text))


def detect_image_mime_type(data: bytes) -> Optional[str]:
    """Detect supported raster image MIME type from magic bytes only."""
    if not data:
        return None
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _max_base64_chars(max_bytes: int) -> int:
    return ((max_bytes + 2) // 3) * 4


def _format_byte_limit(num_bytes: int) -> str:
    mib = 1024 * 1024
    if num_bytes >= mib and num_bytes % mib == 0:
        return f"{num_bytes // mib} MiB"
    return f"{num_bytes} bytes"


def _normalize_url_hostname(raw_host: str, *, label: str) -> str:
    """Normalize a URL hostname (incl. Unicode/NFKC/UTS#46 dots & IDNA) to ASCII.

    Prevents IDNA/fullwidth/ideographic-dot SSRF bypasses such as
    ``169.254.169.254\\u3002`` or fullwidth ``localhost.local``.
    """
    normalized = unicodedata.normalize("NFKC", raw_host).translate(_UNICODE_DOTS_TABLE)
    normalized = normalized.rstrip(".")
    if not normalized:
        raise ValueError(f"{label}: image URL is missing a hostname.")

    # IPv6 literals contain ':' and must not go through IDNA label encoding.
    if ":" in normalized:
        return normalized.lower()

    if not normalized.isascii():
        try:
            normalized = normalized.encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise ValueError(f"{label}: image URL has an invalid internationalized hostname.") from exc

    host = normalized.rstrip(".").lower()
    if not host:
        raise ValueError(f"{label}: image URL is missing a hostname.")
    return host


def _is_public_global_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True only for globally routable public IP addresses."""
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return _is_public_global_ip(ip.ipv4_mapped)
        if ip in _IPV4_TRANSLATED_NETWORK:
            embedded = ipaddress.IPv4Address(ip.packed[-4:])
            return _is_public_global_ip(embedded)
        if (
            not ip.is_global
            or ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return False
        return not any(ip in net for net in _BLOCKED_IPV6_NETWORKS)

    if (
        not ip.is_global
        or ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return False
    return not any(ip in net for net in _BLOCKED_IPV4_NETWORKS)


def _parse_ip_literal(host: str) -> Optional[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Parse an IPv4/IPv6 literal (stripping any %zone suffix), or return None."""
    cleaned = host.split("%", 1)[0]
    try:
        return ipaddress.ip_address(cleaned)
    except ValueError:
        return None


def _looks_like_obfuscated_ip(host: str) -> bool:
    """Catch integer/hex/octal IPv4 obfuscations like 2130706433 or 0177.0.0.1."""
    labels = [label for label in host.split(".") if label]
    if not labels:
        return True
    for label in labels:
        low = label.lower()
        if low.startswith("0x"):
            try:
                int(low, 16)
                continue
            except ValueError:
                return False
        if low.isdigit():
            continue
        return False
    return True


def validate_https_image_url(url: str, *, label: str = "image") -> str:
    """Validate a pass-through public https:// image URL with SSRF guards."""
    if len(url) > MAX_IMAGE_URL_LENGTH:
        raise ValueError(
            f"{label}: image URL exceeds maximum length ({MAX_IMAGE_URL_LENGTH} chars)."
        )
    if any(ord(ch) <= 32 or ord(ch) == 127 for ch in url) or "\\" in url:
        raise ValueError(
            f"{label}: image URL must not contain whitespace, backslashes, or control characters."
        )

    try:
        parsed = urlsplit(url)
    except ValueError as exc:
        raise ValueError(f"{label}: malformed image URL.") from exc

    scheme = (parsed.scheme or "").lower()
    if scheme != "https":
        raise ValueError(
            f"{label}: only public 'https://' image URLs are allowed "
            "(or pass a local image file path or base64 data:image/... URL)."
        )
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{label}: image URL must not embed credentials.")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError(f"{label}: image URL has an invalid port.") from exc

    raw_host = parsed.hostname
    if not raw_host:
        raise ValueError(f"{label}: image URL is missing a hostname.")

    host = _normalize_url_hostname(raw_host, label=label)
    if host in _BLOCKED_HOSTNAMES or host.endswith(_BLOCKED_HOST_SUFFIXES):
        raise ValueError(
            f"{label}: image URL targets a blocked local or internal hostname."
        )

    ip = _parse_ip_literal(host)
    if ip is not None:
        if not _is_public_global_ip(ip):
            raise ValueError(
                f"{label}: image URL targets a private, loopback, link-local, or reserved IP."
            )
        return url

    if _looks_like_obfuscated_ip(host):
        raise ValueError(
            f"{label}: image URL uses a non-standard numeric IP host."
        )
    if "." not in host:
        raise ValueError(
            f"{label}: image URL must use a public fully-qualified domain name."
        )

    # Best-effort DNS check: if local resolution succeeds and yields any
    # non-global IP, block it. Offline/sandbox gaierror is tolerated because
    # Herald forwards the URL rather than opening an outbound socket itself.
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        infos = []
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        resolved_ip = _parse_ip_literal(str(sockaddr[0]))
        if resolved_ip is not None and not _is_public_global_ip(resolved_ip):
            raise ValueError(
                f"{label}: image URL resolves to a private, loopback, or reserved IP."
            )

    return url


def validate_image_data_url(
    url: str,
    *,
    label: str = "image",
    max_image_bytes: int = MAX_IMAGE_BYTES,
) -> Tuple[str, int]:
    """Validate an inline ``data:image/...;base64,...`` URL without leaking payloads.

    Returns ``(canonical_data_url, raw_byte_length)``.
    """
    header, sep, payload = url.partition(",")
    if not sep:
        raise ValueError(f"{label}: malformed data: URL (missing comma separator).")

    meta = header[5:]  # strip "data:"
    tokens = [tok.strip().lower() for tok in meta.split(";") if tok.strip()]
    if len(tokens) < 2 or tokens[-1] != "base64":
        raise ValueError(
            f"{label}: data: URL must be base64-encoded (data:image/<type>;base64,...)."
        )

    declared = tokens[0]
    if declared == "image/jpg":
        declared = "image/jpeg"
    if declared not in SUPPORTED_IMAGE_MIME_TYPES:
        safe_declared = re.sub(r"[^a-z0-9/+. -]", "", declared)[:40] or "unknown"
        raise ValueError(
            f"{label}: unsupported image MIME type '{safe_declared}'. "
            "Supported formats: image/png, image/jpeg, image/gif, image/webp."
        )

    if not payload or not payload.strip():
        raise ValueError(f"{label}: data: URL image payload is empty.")

    # Pre-decode size gate to prevent allocating giant buffers on huge inputs.
    max_b64 = _max_base64_chars(max_image_bytes)
    limit_text = _format_byte_limit(max_image_bytes)
    # Generous upper bound allowing MIME-style line wrapping before compacting.
    if len(payload) > max_b64 + (max_b64 // 32) + 128:
        raise ValueError(
            f"{label}: image exceeds per-image size limit (max {limit_text} raw)."
        )

    compact = re.sub(r"[\r\n\t ]+", "", payload)
    if not compact:
        raise ValueError(f"{label}: data: URL image payload is empty.")
    if len(compact) > max_b64:
        raise ValueError(
            f"{label}: image exceeds per-image size limit (max {limit_text} raw)."
        )

    try:
        decoded = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{label}: invalid base64 payload in data: URL.") from exc

    if not decoded:
        raise ValueError(f"{label}: data: URL image payload is empty.")
    if len(decoded) > max_image_bytes:
        raise ValueError(
            f"{label}: image exceeds per-image size limit "
            f"({len(decoded)} bytes > {max_image_bytes} bytes)."
        )

    detected = detect_image_mime_type(decoded)
    if detected is None:
        raise ValueError(
            f"{label}: data: URL bytes are not a valid PNG, JPEG, GIF, or WebP image."
        )
    if detected != declared:
        raise ValueError(
            f"{label}: data: URL declared '{declared}' but image bytes are '{detected}'."
        )

    return f"data:{detected};base64,{compact}", len(decoded)


def _is_sensitive_path(path: Path) -> bool:
    """Return True if path targets well-known secret/credential or kernel-pseudo locations."""
    if len(path.parts) >= 2 and path.parts[0] == os.sep and path.parts[1].lower() in {"proc", "sys", "dev"}:
        return True
    name = path.name.lower()
    if name in _SENSITIVE_FILENAMES or name.startswith(".env."):
        return True
    if path.suffix.lower() in _SENSITIVE_SUFFIXES:
        return True
    for part in path.parts:
        if part.lower() in _SENSITIVE_DIR_PARTS:
            return True
    return False


def load_local_image_as_data_url(
    path_str: str,
    *,
    label: str = "image",
    max_image_bytes: int = MAX_IMAGE_BYTES,
) -> Tuple[str, int]:
    """Read, validate, and base64-encode a local image file safely.

    Returns ``(data_url, raw_byte_length)``.
    """
    if "\x00" in path_str or any(ord(ch) < 32 or ord(ch) == 127 for ch in path_str):
        raise ValueError(f"{label}: image path contains invalid control characters.")

    raw_path = Path(path_str)
    if ".." in raw_path.parts:
        raise ValueError(
            f"{label}: image path must not contain '..' traversal segments."
        )

    expanded = Path(os.path.expanduser(path_str))
    if ".." in expanded.parts:
        raise ValueError(
            f"{label}: image path must not contain '..' traversal segments."
        )

    resolved = expanded.resolve(strict=False)
    if _is_sensitive_path(expanded) or _is_sensitive_path(resolved):
        raise ValueError(
            f"{label}: reading credential or sensitive files as images is not allowed."
        )

    open_flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(os.fspath(resolved), open_flags)
    except FileNotFoundError as exc:
        raise ValueError(
            f"{label}: image file not found (pass an existing local image path, "
            "a base64 data:image/... URL, or a public https:// URL)."
        ) from exc
    except IsADirectoryError as exc:
        raise ValueError(f"{label}: image path must be a regular file, not a directory.") from exc
    except PermissionError as exc:
        raise ValueError(f"{label}: permission denied reading image file.") from exc
    except OSError as exc:
        raise ValueError(f"{label}: cannot open image file ({exc.strerror or 'OS error'}).") from exc

    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ValueError(
                f"{label}: image path must be a regular file (devices, pipes, "
                "sockets, and directories are rejected)."
            )
        if st.st_size > max_image_bytes:
            raise ValueError(
                f"{label}: image exceeds per-image size limit "
                f"({st.st_size} bytes > {max_image_bytes} bytes)."
            )

        chunks: List[bytes] = []
        total = 0
        with os.fdopen(fd, "rb") as handle:
            fd = -1  # ownership transferred to handle
            while True:
                remaining = (max_image_bytes - total) + 1
                chunk = handle.read(min(_READ_CHUNK_BYTES, remaining))
                if not chunk:
                    break
                total += len(chunk)
                if total > max_image_bytes:
                    raise ValueError(
                        f"{label}: image exceeds per-image size limit "
                        f"(max {max_image_bytes} bytes)."
                    )
                chunks.append(chunk)
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass

    data = b"".join(chunks)
    if not data:
        raise ValueError(f"{label}: image file is empty.")

    detected = detect_image_mime_type(data)
    if detected is None:
        raise ValueError(
            f"{label}: file is not a valid PNG, JPEG, GIF, or WebP image."
        )

    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{detected};base64,{encoded}", len(data)


def _normalize_image_source(
    source: Any,
    *,
    label: str,
    max_image_bytes: int,
) -> Tuple[str, int]:
    """Validate and resolve one image reference (data: URL, https:// URL, or local path).

    Returns ``(wire_url, raw_byte_size)`` where ``raw_byte_size`` is 0 for
    pass-through ``https://`` URLs and the decoded byte count for inline/local
    images.
    """
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"{label} must be a non-empty image path or URL string.")
    s = source.strip()

    if s[:5].lower() == "data:":
        return validate_image_data_url(s, label=label, max_image_bytes=max_image_bytes)

    if not _WINDOWS_DRIVE_RE.match(s):
        scheme_match = _URL_SCHEME_RE.match(s)
        if scheme_match:
            scheme = scheme_match.group(1).lower()
            if scheme == "https":
                return validate_https_image_url(s, label=label), 0
            if scheme == "http":
                raise ValueError(
                    f"{label}: plaintext 'http://' image URLs are not allowed; "
                    "use 'https://', a data:image/...;base64,... URL, or a local file path."
                )
            raise ValueError(
                f"{label}: unsupported image URL scheme '{scheme}'; "
                "use 'https://', a data:image/...;base64,... URL, or a local file path."
            )

    return load_local_image_as_data_url(s, label=label, max_image_bytes=max_image_bytes)


def _normalize_content_part(
    part: Any,
    *,
    label: str,
    max_image_bytes: int,
) -> Tuple[Dict[str, Any], int, int]:
    """Normalize one OpenAI content part.

    Returns ``(normalized_part, image_count_delta, image_bytes_delta)``.
    """
    if not isinstance(part, dict):
        raise ValueError(f"{label} must be a content-part object.")
    ptype = part.get("type")
    if ptype == "text":
        extra_keys = set(part.keys()) - {"type", "text"}
        if extra_keys:
            raise ValueError(
                f"{label}: unexpected keys on text part: {', '.join(sorted(extra_keys))}."
            )
        text = part.get("text")
        if not isinstance(text, str):
            raise ValueError(f"{label}.text must be a string.")
        return {"type": "text", "text": text}, 0, 0

    if ptype == "image_url":
        extra_keys = set(part.keys()) - {"type", "image_url"}
        if extra_keys:
            raise ValueError(
                f"{label}: unexpected keys on image_url part: {', '.join(sorted(extra_keys))}."
            )
        image_obj = part.get("image_url")
        if not isinstance(image_obj, dict):
            raise ValueError(
                f"{label}.image_url must be an object with 'url' (and optional 'detail')."
            )
        extra_img_keys = set(image_obj.keys()) - {"url", "detail"}
        if extra_img_keys:
            raise ValueError(
                f"{label}.image_url has unexpected keys: {', '.join(sorted(extra_img_keys))}."
            )
        detail = image_obj.get("detail")
        if detail is not None:
            if not isinstance(detail, str) or detail not in ALLOWED_IMAGE_DETAILS:
                raise ValueError(
                    f"{label}.image_url.detail must be one of: auto, low, high."
                )
        wire_url, raw_bytes = _normalize_image_source(
            image_obj.get("url"),
            label=f"{label}.image_url.url",
            max_image_bytes=max_image_bytes,
        )
        normalized_img: Dict[str, Any] = {"url": wire_url}
        if detail is not None:
            normalized_img["detail"] = detail
        return {"type": "image_url", "image_url": normalized_img}, 1, raw_bytes

    raise ValueError(
        f"{label}.type must be 'text' or 'image_url' (got {ptype!r})."
    )


def normalize_llm_messages(
    messages: Any,
    images: Any = None,
    *,
    max_images: int = MAX_IMAGES_PER_CALL,
    max_image_bytes: int = MAX_IMAGE_BYTES,
    max_total_image_bytes: int = MAX_TOTAL_IMAGE_BYTES,
) -> List[Dict[str, Any]]:
    """Validate and normalize chat messages + optional top-level ``images``.

    Always returns fresh dicts/lists without mutating caller inputs.
    Plain-text messages remain plain strings when no images are attached.
    Raises ``ValueError`` with a safe, base64-free message on any invalid input.
    """
    if not isinstance(messages, list) or not messages:
        raise ValueError("'messages' is required (list of {role, content} objects).")

    if images is not None and not isinstance(images, list):
        raise ValueError("'images' must be a list of image paths or URLs when provided.")

    normalized: List[Dict[str, Any]] = []
    total_images = 0
    total_image_bytes = 0
    last_user_index: Optional[int] = None

    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{idx}] must be an object with role and content.")
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(
                f"messages[{idx}].role must be system, user, or assistant."
            )
        if role == "user":
            last_user_index = idx

        content = message.get("content")
        if isinstance(content, str):
            normalized.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            if role != "user":
                raise ValueError(
                    f"messages[{idx}].content must be a string for role='{role}' "
                    "(multimodal content arrays are supported only on role='user')."
                )
            if not content:
                raise ValueError(
                    f"messages[{idx}].content must be a non-empty array of content parts."
                )
            if len(content) > MAX_CONTENT_PARTS_PER_MESSAGE:
                raise ValueError(
                    f"messages[{idx}].content exceeds maximum parts per message "
                    f"({MAX_CONTENT_PARTS_PER_MESSAGE})."
                )
            parts: List[Dict[str, Any]] = []
            for p_idx, raw_part in enumerate(content):
                part, img_delta, byte_delta = _normalize_content_part(
                    raw_part,
                    label=f"messages[{idx}].content[{p_idx}]",
                    max_image_bytes=max_image_bytes,
                )
                if img_delta:
                    total_images += img_delta
                    if total_images > max_images:
                        raise ValueError(
                            f"Too many images in request (max {max_images} per call)."
                        )
                    total_image_bytes += byte_delta
                    if total_image_bytes > max_total_image_bytes:
                        raise ValueError(
                            f"Total image payload exceeds limit "
                            f"({total_image_bytes} bytes > {max_total_image_bytes} bytes)."
                        )
                parts.append(part)
            normalized.append({"role": role, "content": parts})
            continue

        raise ValueError(
            f"messages[{idx}].content must be a string"
            + (" or array of content parts." if role == "user" else ".")
        )

    if images:
        if last_user_index is None:
            raise ValueError("'images' requires at least one message with role='user'.")
        if total_images + len(images) > max_images:
            raise ValueError(f"Too many images in request (max {max_images} per call).")

        attached_parts: List[Dict[str, Any]] = []
        for img_idx, raw_img in enumerate(images):
            wire_url, raw_bytes = _normalize_image_source(
                raw_img,
                label=f"images[{img_idx}]",
                max_image_bytes=max_image_bytes,
            )
            total_images += 1
            total_image_bytes += raw_bytes
            if total_image_bytes > max_total_image_bytes:
                raise ValueError(
                    f"Total image payload exceeds limit "
                    f"({total_image_bytes} bytes > {max_total_image_bytes} bytes)."
                )
            attached_parts.append({
                "type": "image_url",
                "image_url": {"url": wire_url},
            })

        target = normalized[last_user_index]
        existing_content = target["content"]
        if isinstance(existing_content, str):
            merged: List[Dict[str, Any]] = []
            if existing_content:
                merged.append({"type": "text", "text": existing_content})
            merged.extend(attached_parts)
        else:
            merged = list(existing_content) + attached_parts

        if len(merged) > MAX_CONTENT_PARTS_PER_MESSAGE:
            raise ValueError(
                f"messages[{last_user_index}].content exceeds maximum parts per message "
                f"({MAX_CONTENT_PARTS_PER_MESSAGE}) after attaching images."
            )
        target["content"] = merged

    return normalized


def extract_text_from_choice_message(message_obj: Any) -> str:
    """Extract text from an OpenAI choice message (string or text-part list)."""
    if not isinstance(message_obj, dict):
        return ""
    content = message_obj.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = [
            part["text"]
            for part in content
            if isinstance(part, dict)
            and part.get("type") == "text"
            and isinstance(part.get("text"), str)
        ]
        return "".join(chunks)
    return ""

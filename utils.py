import re
from urllib.parse import urlparse
import tldextract
import math
from collections import Counter
import ipaddress  # new: for robust IP (v4 + v6) detection

# Simple IPv4 regex (kept from your original)
IP_RE = re.compile(r'^(?:\d{1,3}\.){3}\d{1,3}$')


def shannon_entropy(s: str) -> float:
    """Calculate the Shannon entropy of a string."""
    if not s:
        return 0.0
    probabilities = Counter(s)
    total_length = len(s)
    entropy = 0.0
    for count in probabilities.values():
        p = count / total_length
        entropy -= p * math.log2(p)
    return entropy


def extract_url_features(url: str) -> dict:
    """
    Extract handcrafted features from a URL.
    This version is defensive: it won't crash on malformed or IPv6 URLs.
    """
    # Basic type and emptiness checks
    if not isinstance(url, str):
        return {}
    url = url.strip()
    if not url:
        return {}

    original_url = url  # keep original for length/flags

    # Ensure urlparse always sees a scheme
    if "://" not in url:
        url_for_parse = "http://" + url
    else:
        url_for_parse = url

    # Path-confusion feature: count '//' after the protocol
    protocol_end_index = url_for_parse.find("://")
    if protocol_end_index != -1:
        path_start_index = protocol_end_index + 3
        count_double_slash = url_for_parse[path_start_index:].count("//")
    else:
        count_double_slash = url_for_parse.count("//")

    # Parse URL safely
    try:
        parsed = urlparse(url_for_parse)
    except Exception:
        # If parsing totally fails, bail out gracefully
        return {}

    domain = parsed.netloc or parsed.path or ""

    # Remove credentials if present (user:pass@host)
    if "@" in domain:
        domain = domain.split("@", 1)[1]

    host = domain.lower()
    path = parsed.path or ""
    query = parsed.query or ""

    # --- Safe TLD extraction (tldextract can choke on some weird/IPv6 hosts) ---
    te_domain = ""
    te_suffix = ""
    if host:
        try:
            te = tldextract.extract(host)
            te_domain = te.domain or ""
            te_suffix = te.suffix or ""
        except Exception:
            # On failure, just leave them empty; don't crash
            te_domain = ""
            te_suffix = ""

    # --- IP detection (IPv4 + IPv6) ---
    has_ip = 0
    if host:
        host_for_ip = host.strip("[]")  # IPv6 often represented as [::1]
        try:
            ipaddress.ip_address(host_for_ip)
            has_ip = 1
        except ValueError:
            # Fallback simple IPv4 regex
            if IP_RE.match(host_for_ip):
                has_ip = 1

    # --- Feature construction ---
    features: dict = {}

    features["url_len"] = len(original_url)
    features["host_len"] = len(host)
    features["path_len"] = len(path)
    features["count_dots"] = host.count(".")
    features["count_hyphens"] = host.count("-") + path.count("-")
    features["has_at"] = 1 if "@" in original_url else 0
    features["has_ip"] = has_ip
    features["has_query"] = 1 if query else 0
    features["num_subdirs"] = path.count("/")
    features["is_https"] = 1 if parsed.scheme == "https" else 0
    features["sld_len"] = len(te_domain)
    features["tld"] = te_suffix

    # --- Extra phishing / lookalike features ---

    # 1. Non-ASCII characters → possible IDN/homograph
    features["has_non_ascii"] = 1 if any(ord(c) > 127 for c in original_url) else 0

    # 2. Extra double slashes in the path
    features["count_double_slash"] = count_double_slash

    # 3. Punycode (IDN) → xn-- prefix
    features["has_punycode"] = 1 if host.startswith("xn--") else 0

    # 4. Entropy of host (high → random-looking, DGA-ish)
    features["host_entropy"] = shannon_entropy(host)

    # 5. Non-standard port used
    features["has_port"] = (
        1 if (parsed.netloc and ":" in parsed.netloc and not parsed.netloc.endswith(":"))
        else 0
    )

    return features

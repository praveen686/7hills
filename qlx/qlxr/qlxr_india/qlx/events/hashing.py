"""Rolling hash chain for BRAHMASTRA event log tamper evidence.

Optional layer: if enabled, each JSONL line's hash incorporates the
previous line's hash, forming a chain.  Any modification to a past
event breaks the chain from that point forward.

Algorithm: SHA-256(prev_hash || line_bytes)
Initial hash: SHA-256(b"BRAHMASTRA_GENESIS")

This is NOT cryptographic signing — it's tamper *evidence*, not
tamper *proof*.  An attacker with write access can recompute the
chain.  But it detects accidental corruption, truncation, and
out-of-order edits during forensic review.
"""

from __future__ import annotations

import hashlib

GENESIS = hashlib.sha256(b"BRAHMASTRA_GENESIS").hexdigest()


def chain_hash(prev_hash: str, line: str) -> str:
    """Compute the next hash in the chain.

    Parameters
    ----------
    prev_hash : str
        Hex-encoded SHA-256 of the previous line's chain hash.
        Use GENESIS for the first line.
    line : str
        The JSONL line (without trailing newline).

    Returns
    -------
    str
        Hex-encoded SHA-256 hash.
    """
    data = prev_hash.encode("ascii") + b"||" + line.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def verify_chain(lines: list[str], hashes: list[str]) -> tuple[bool, int]:
    """Verify a sequence of lines against their chain hashes.

    Parameters
    ----------
    lines : list[str]
        JSONL lines (no trailing newlines).
    hashes : list[str]
        Expected chain hashes (one per line).

    Returns
    -------
    (valid, break_index)
        valid=True if entire chain is consistent.
        If valid=False, break_index is the first inconsistent line.
    """
    if len(lines) != len(hashes):
        return False, min(len(lines), len(hashes))

    prev = GENESIS
    for i, (line, expected) in enumerate(zip(lines, hashes)):
        computed = chain_hash(prev, line)
        if computed != expected:
            return False, i
        prev = computed

    return True, len(lines)


def compute_chain(lines: list[str]) -> list[str]:
    """Compute the full hash chain for a list of lines.

    Returns list of hex-encoded SHA-256 hashes, one per line.
    """
    result: list[str] = []
    prev = GENESIS
    for line in lines:
        h = chain_hash(prev, line)
        result.append(h)
        prev = h
    return result

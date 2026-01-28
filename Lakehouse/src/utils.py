import hashlib
import json
import logging
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

LOGGER = logging.getLogger(__name__)


def ensure_directories(paths: Sequence[Path]) -> None:
    """Create directories if they do not exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def compute_file_hash_sha256(path: Path, chunk_size: int = 8192) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def canonical_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def compute_chunk_config_id(config: Dict[str, Any]) -> str:
    canonical = canonical_json_dumps(config)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def generate_run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def detect_repeated_lines(
    pages_raw_texts: Sequence[str], top_k: int = 2, bottom_k: int = 2
) -> Set[str]:
    """Return header/footer lines that repeat across many pages."""
    if len(pages_raw_texts) < 3:
        return set()
    line_counts: Dict[str, int] = {}
    for text in pages_raw_texts:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        head = lines[:top_k]
        tail = lines[-bottom_k:] if bottom_k else []
        for ln in head + tail:
            line_counts[ln] = line_counts.get(ln, 0) + 1
    threshold = max(2, int(len(pages_raw_texts) * 0.6))
    return {ln for ln, count in line_counts.items() if count >= threshold}


def clean_text(raw_text: str, repeated_lines: Optional[Set[str]] = None) -> str:
    """Clean raw PDF page text following the prescribed data-cleaning pipeline.

    Pipeline (must keep order):
      1) Ligatures -> NFKC -> drop replacement/format chars
      2) Remove separators/page numbers + repeated_lines
      3) Remove figure/chart noise lines (token-length heuristics)
      4) Hyphenation repair (only hyphen directly before newline)
      5) Footnote marker spacing ("1As" -> "1 As")
      6) Word unglueing (punctuation spacing + CamelCase + space collapse)
      7) Smooth line breaks, keep paragraphs, normalize whitespace
    """

    def _drop_cf_and_replacement(text: str) -> str:
        return "".join(
            ch
            for ch in text
            if ch != "\ufffd" and unicodedata.category(ch) != "Cf"
        )

    def _is_separator_line(line: str) -> bool:
        return bool(re.fullmatch(r"[-_]{10,}", line))

    def _is_page_number_line(line: str) -> bool:
        return bool(re.fullmatch(r"\d+", line))

    def _is_figure_noise(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        # Single-character lines (typical axis ticks / legend artifacts).
        if re.fullmatch(r"[A-Za-z0-9%()\\.]{1,2}", stripped):
            return True
        tokens = [tok for tok in re.split(r"\s+", stripped) if tok]
        if not tokens:
            return False
        len1_ratio = sum(1 for tok in tokens if len(tok) == 1) / len(tokens)
        if len(tokens) >= 3 and len1_ratio >= 0.6:
            return True
        short_ratio = sum(1 for tok in tokens if len(tok) <= 2) / len(tokens)
        if len(tokens) >= 6 and short_ratio >= 0.8:
            return True
        if len(tokens) >= 3 and short_ratio >= 0.9 and len(stripped) <= 40:
            return True
        return False

    repeated_lines = repeated_lines or set()

    # 1) Ligatures -> NFKC -> drop replacement/format chars
    text = raw_text.replace("\r\n", "\n").replace("\u00a0", " ")
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = unicodedata.normalize("NFKC", text)
    text = (
        text.replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2212", "-")
        .replace("\u00ad", "")
    )
    text = _drop_cf_and_replacement(text)

    # 2) Remove header/footer/separator/page-number lines and repeated lines
    filtered_lines: List[str] = []
    for ln in text.splitlines():
        stripped = ln.strip()
        if not stripped:
            filtered_lines.append("")
            continue
        if stripped in repeated_lines:
            continue
        if _is_separator_line(stripped):
            continue
        if _is_page_number_line(stripped):
            continue
        filtered_lines.append(stripped)

    # 3) Remove figure/chart noise lines (after header/footer removal)
    noise_filtered: List[str] = []
    for ln in filtered_lines:
        if not ln:
            noise_filtered.append("")
            continue
        if _is_figure_noise(ln):
            continue
        noise_filtered.append(ln)

    text = "\n".join(noise_filtered)

    # 4) Hyphenation repair:
    #    a) join when a hyphen sits right before a newline
    #    b) join when a hyphen is followed by whitespace + lowercase start (newline may
    #       have been flattened earlier by the extractor)
    text = re.sub(r"(\w+)[-\u2010\u2011\u2012\u2013\u2212]\s*\n(\w+)", r"\1\2", text)
    text = re.sub(r"(\w+)[-\u2010\u2011\u2012\u2013\u2212]\s+([a-z]{2,})", r"\1\2", text)

    # 5) Footnote marker spacing
    text = re.sub(r"(^|\s)(\d+)([A-Z])", r"\1\2 \3", text)

    # 6) Word unglueing: punctuation spacing, CamelCase spacing, collapse spaces
    text = re.sub(r"([.!?])(?=[A-Z])", r"\1 ", text)
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r" {2,}", " ", text)

    # 7) Smooth line breaks, keep paragraphs, normalize whitespace/newlines
    paragraphs: List[str] = []
    for block in re.split(r"\n\n+", text):
        block = block.strip()
        if not block:
            continue
        block = re.sub(r"\s*\n\s*", " ", block)
        block = re.sub(r"\s{2,}", " ", block)
        paragraphs.append(block)
    merged = "\n\n".join(paragraphs)
    merged = re.sub(r"\n{3,}", "\n\n", merged)
    return merged.strip()


def _find_split_point(text: str, start: int, target: int, max_len: int) -> int:
    """Heuristic split point preferring paragraphs then sentences."""
    window = text[start : min(len(text), start + max_len)]
    rel_target = min(len(window), target)
    preferred = ["\n\n", ". ", "! ", "? ", "\n"]
    best = start + rel_target
    for sep in preferred:
        idx = window.rfind(sep, 0, rel_target + 200)
        if idx != -1 and start + idx > best - 200:
            best = start + idx + len(sep)
            break
    return max(best, start + rel_target)


def chunk_text_with_overlap(
    text: str, config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    if not text:
        return []
    target = int(config.get("target_chars", 3000))
    overlap_chars = int(config.get("overlap_chars", 450))
    overlap_sentences_cfg = config.get("overlap_sentences", None)
    if target <= 0:
        raise ValueError("target_chars must be > 0")

    # Sentence span detection (greedy, simple rule: end on . ! ? followed by space/newline or EOS).
    sentence_spans: List[Tuple[int, int]] = []
    start = 0
    pattern = re.compile(r"[.!?](?=\s|\n|$)")
    for m in pattern.finditer(text):
        end = m.end()
        # extend to consume trailing spaces/newlines after the punctuation
        while end < len(text) and text[end].isspace():
            end += 1
        if end > start:
            sentence_spans.append((start, end))
            start = end
    if start < len(text):
        sentence_spans.append((start, len(text)))

    if not sentence_spans:
        return []

    if overlap_sentences_cfg is not None:
        overlap_sents = max(0, int(overlap_sentences_cfg))
    else:
        # derive from overlap_chars if configured
        avg_len = sum(e - s for s, e in sentence_spans) / max(1, len(sentence_spans))
        if overlap_chars > 0 and avg_len > 0:
            overlap_sents = max(0, int(round(overlap_chars / avg_len)))
        else:
            overlap_sents = 0

    chunks: List[Dict[str, Any]] = []
    idx = 0
    chunk_index = 0
    max_len = int(target * 1.3)
    n_sent = len(sentence_spans)

    while idx < n_sent:
        accum_start = sentence_spans[idx][0]
        accum_end = sentence_spans[idx][1]
        j = idx
        while j + 1 < n_sent and (accum_end - accum_start) < target:
            next_end = sentence_spans[j + 1][1]
            if next_end - accum_start > max_len:
                break
            j += 1
            accum_end = sentence_spans[j][1]

        chunk_text = text[accum_start:accum_end].strip()
        if chunk_text:
            min_size = max(50, int(target * 0.4))
            if chunks and len(chunk_text) < min_size and j + 1 >= n_sent:
                prev = chunks[-1]
                prev["chunk_text"] = (prev["chunk_text"] + " " + chunk_text).strip()
                prev["char_end"] = accum_end
            else:
                chunks.append(
                    {
                        "chunk_index": chunk_index,
                        "chunk_text": chunk_text,
                        "char_start": accum_start,
                        "char_end": accum_end,
                    }
                )
                chunk_index += 1

        next_idx = j + 1
        if overlap_sents and next_idx > overlap_sents:
            next_idx = max(idx + 1, j + 1 - overlap_sents)
        idx = next_idx

    return chunks


def stable_chunk_id(
    doc_id: int,
    chunk_config_id: str,
    chunk_index: int,
    page_start: int,
    page_end: int,
    char_start: int,
    char_end: int,
    chunk_text: str,
) -> str:
    payload = (
        f"{doc_id}|{chunk_config_id}|{chunk_index}|"
        f"{page_start}-{page_end}|{char_start}-{char_end}|"
        f"{hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_metadata(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "file_size_bytes": stat.st_size,
        "modified_time": datetime.fromtimestamp(stat.st_mtime),
    }


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

"""Extract structured failure information from a notebook test log.

The log is the combined stdout/stderr of ``python -m treon --verbose test_x.ipynb`` (treon fork that logs
``Start executing cell <N>`` before every code cell) followed by a few lines printed by validate_notebooks.py.

Stdlib only. None of the public functions raise, whatever the input: on unexpected input they fall back to
partial results.
"""

import bisect
import hashlib
import re
from typing import NamedTuple, Optional

from .constants import (
    CELL_SOURCE_MAX_LINES,
    EXCERPT_MAX_BYTES,
    LOG_TAIL_MAX_LINES,
    MESSAGE_MAX_CHARS,
    TRACEBACK_MAX_LINES,
    ErrorClass,
    FailureCategory,
    Status,
)

RESULT_KEYS = (
    "category",
    "cell_index",
    "cell_source_excerpt",
    "exception_type",
    "exception_message",
    "traceback_excerpt",
    "log_tail",
)

NORMALIZED_MESSAGE_MAX_CHARS = 300
SIGNATURE_LENGTH = 12

_TYPE_MAX_CHARS = 256
# Lines are truncated before regex matching to keep the parser linear on pathological input.
_MATCH_MAX_CHARS = 4096
_NORMALIZE_INPUT_MAX_CHARS = 20000

_ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[()#][0-9A-Za-z]|[@-Z\\-_])")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_CELL_START_RE = re.compile(r"Start executing cell\s+(\d{1,9})\b")
_ERROR_IN_TESTING = "ERROR in testing"
_CELL_ERROR_HEADER = "An error occurred while executing the following cell"
_SEPARATOR = "-" * 18
_STREAM_HEADER_RE = re.compile(r"----- [\w.-]+ -----")
_LONG_DASH_RE = re.compile(r"-{20,}")
_TB_HEADER = "Traceback (most recent call last)"
_DEAD_KERNEL_RE = re.compile(r"DeadKernelError\s*:\s*(.*)")
_TIMEOUT_LINE_RE = re.compile(r"timeout reached \(", re.IGNORECASE)
_LOG_PREFIX_RE = re.compile(r"^\[[\w.-]+\]\s*")

# Lines printed after treon by treon's summary / validate_notebooks.py; they terminate an error block.
_NOISE_RE = re.compile(
    r"\[validate_notebooks\]"
    r"|WARNING: Package\(s\) not found"
    r"|OpenVINO[\w ]* (?:before|after) notebook execution:"
    r"|[\w.-]+ is missing in validation environment\."
    r"|Notebook test \[.*\] timeout reached"
    r"|ERROR in testing "
)

_EXC_LINE_RE = re.compile(r"([A-Za-z_][\w.]*)(?:\s*:\s*(.*))?")
_EXC_SUFFIX_RE = re.compile(r"[A-Za-z_]\w*(?:Error|Exception|Exit|Interrupt|Failure|Warning|Iteration|Timeout|Fault|Abort)")
_KNOWN_EXC_NAMES = frozenset(
    {
        "Exception",
        "BaseException",
        "KeyboardInterrupt",
        "StopIteration",
        "StopAsyncIteration",
        "GeneratorExit",
        "SystemExit",
        "IncompleteRead",
        "RemoteDisconnected",
        "gaierror",
        "herror",
        "timeout",
    }
)
_GENERIC_EXC_SUFFIXES = ("Error", "Exception", "Exit", "Interrupt", "Failure", "Warning")
_GENERIC_EXC_RE = re.compile(r"^\s*([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Failure|Warning))\s*:\s*(.*)$")
_ALL_CAPS_TOKEN_RE = re.compile(r"^\s*[A-Z][A-Z0-9_]*:")
_HINT_RE = re.compile(r"^\s*([A-Za-z_][\w.]*)\s*:\s*(.*)$", re.DOTALL)

_WIN_PATH_RE = re.compile(r"(?<![\w])[A-Za-z]:[\\/][^\s'\"`<>|:,;()\[\]{}]*")
_POSIX_PATH_RE = re.compile(r"(?<![\w.:/~\\-])/[^\s'\"`<>|:,;()\[\]{}]+")
_UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
_HEX_PREFIXED_RE = re.compile(r"\b0[xX][0-9a-fA-F]+\b")
_LONG_HEX_RE = re.compile(r"\b(?=[0-9a-fA-F]*\d)[0-9a-fA-F]{12,}\b")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)*")
_WHITESPACE_RE = re.compile(r"\s+")

_DISK_RE = re.compile(r"No space left on device|Errno 28\b")
_MEMORY_TYPES = frozenset({"MemoryError", "OutOfMemoryError"})
_MEMORY_RE = re.compile(r"(?i:out of memory|std::bad_alloc|cannot allocate memory|unable to allocate|oom-kill)|\bOOM(?:Killed)?\b")
_DEPENDENCY_TYPES = frozenset({"ModuleNotFoundError", "ImportError", "PackageNotFoundError"})
_DEPENDENCY_RE = re.compile(
    r"No matching distribution found|ResolutionImpossible|Could not find a version that satisfies|subprocess-exited-with-error|DLL load failed"
)
_PIP_RE = re.compile(r"\bpip3?\b")
_NETWORK_TYPES = frozenset(
    {
        "ConnectionError",
        "ConnectTimeout",
        "ReadTimeout",
        "ConnectTimeoutError",
        "ReadTimeoutError",
        "HTTPError",
        "URLError",
        "SSLError",
        "MaxRetryError",
        "ProxyError",
        "HfHubHTTPError",
        "LocalEntryNotFoundError",
        "ChunkedEncodingError",
        "IncompleteRead",
        "RemoteDisconnected",
        "gaierror",
    }
)
_NETWORK_RE = re.compile(
    r"Max retries exceeded|Connection (?:reset|refused|aborted)|Temporary failure in name resolution|Name or service not known"
    r"|Read timed out|Too Many Requests|CERTIFICATE_VERIFY_FAILED"
    r"|Failed to establish a new connection|Network is unreachable|couldn't connect to"
)
# HTTP status codes only count with HTTP context, so that e.g. "index 503 is out of bounds" is not a network error.
_HTTP_STATUS_RE = re.compile(
    r"\b(?:429|502|503|504)\b:?\s*(?:Client Error|Server Error|Service Unavailable|Bad Gateway|Gateway Time-?out|Too Many Requests)"
    r"|(?:\bHTTP(?:/\d(?:\.\d)?)?(?: Error)?|\bstatus(?:[ _]code)?|\bresponse code)\s*[:=]?\s*(?:429|502|503|504)\b"
    r"|<Response \[(?:429|502|503|504)\]>",
    re.IGNORECASE,
)
_OPENVINO_RE = re.compile(r"Exception from src/|\[ GENERAL_ERROR \]|Check '[^\n]+' failed at")


class _Exc(NamedTuple):
    exc_type: str
    message: Optional[str]
    strong: bool


class _CellError(NamedTuple):
    start: int
    end: int
    source_lines: Optional[list]
    traceback_lines: list
    exception: Optional[tuple]


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences (colors, cursor movement, OSC titles)."""
    if text is None:
        return ""
    try:
        if not isinstance(text, str):
            text = _to_str(text)
        return _ANSI_RE.sub("", text).replace("\x1b", "")
    except Exception:
        return ""


def normalize_message(message: Optional[str]) -> str:
    """Remove run-specific details (paths, addresses, ids, numbers) so equal errors compare equal."""
    if not message:
        return ""
    try:
        text = strip_ansi(message)[:_NORMALIZE_INPUT_MAX_CHARS]
        text = _WIN_PATH_RE.sub("<path>", text)
        text = _POSIX_PATH_RE.sub("<path>", text)
        text = _UUID_RE.sub("<uuid>", text)
        text = _HEX_PREFIXED_RE.sub("<hex>", text)
        text = _LONG_HEX_RE.sub("<hex>", text)
        text = _NUMBER_RE.sub("<n>", text)
        text = _WHITESPACE_RE.sub(" ", text).strip()
        return text[:NORMALIZED_MESSAGE_MAX_CHARS]
    except Exception:
        return ""


def compute_signature(category: str, exception_type: Optional[str], exception_message: Optional[str]) -> str:
    """12 hex chars identifying the same error across notebooks, configurations and nights."""
    try:
        payload = f"{category}|{exception_type or ''}|{normalize_message(exception_message)}"
    except Exception:
        payload = "||"
    return hashlib.sha1(payload.encode("utf-8", "replace"), usedforsecurity=False).hexdigest()[:SIGNATURE_LENGTH]


def classify(category: str, exception_type: Optional[str], exception_message: Optional[str], text: str = "") -> str:
    """Heuristic triage class (an ErrorClass value); the first matching rule wins."""
    try:
        return _classify(category, exception_type, exception_message, text)
    except Exception:
        return ErrorClass.OTHER


def parse_log(text: Optional[str], status: str, return_code: Optional[int] = None, error_hint: Optional[str] = None) -> dict:
    """Parse a notebook test log into the failure fields of the nightly report (see RESULT_KEYS)."""
    try:
        return _parse_log(text, status, return_code, error_hint)
    except Exception:
        return _fallback_result(text, status, error_hint)


def analyze_failure(text: Optional[str], status: str, return_code: Optional[int] = None, error_hint: Optional[str] = None) -> dict:
    """parse_log() result extended with "error_class" and "signature"."""
    result = parse_log(text, status, return_code, error_hint)
    result["error_class"] = classify(result["category"], result["exception_type"], result["exception_message"], result["log_tail"] or "")
    result["signature"] = compute_signature(result["category"], result["exception_type"], result["exception_message"])
    return result


def _classify(category, exception_type, exception_message, text):
    if category == FailureCategory.TIMEOUT:
        return ErrorClass.TIMEOUT
    if category == FailureCategory.HARNESS_ERROR:
        return ErrorClass.HARNESS

    exc_type = _last_component(strip_ansi(str(exception_type))) if exception_type else ""
    haystack = f"{exc_type} {strip_ansi(str(exception_message)) if exception_message else ''}"
    if category in (FailureCategory.KERNEL_DIED, FailureCategory.UNKNOWN) and text:
        haystack = f"{haystack}\n{strip_ansi(str(text))}"

    if _DISK_RE.search(haystack):
        return ErrorClass.DISK
    if exc_type in _MEMORY_TYPES or _MEMORY_RE.search(haystack):
        return ErrorClass.MEMORY
    if category == FailureCategory.KERNEL_DIED:
        return ErrorClass.CRASH
    if exc_type in _DEPENDENCY_TYPES or _DEPENDENCY_RE.search(haystack) or (exc_type == "CalledProcessError" and _PIP_RE.search(haystack)):
        return ErrorClass.DEPENDENCY
    if exc_type in _NETWORK_TYPES or _NETWORK_RE.search(haystack) or _HTTP_STATUS_RE.search(haystack):
        return ErrorClass.NETWORK
    if _OPENVINO_RE.search(haystack):
        return ErrorClass.OPENVINO
    if exc_type == "AssertionError":
        return ErrorClass.ASSERTION
    return ErrorClass.OTHER


def _parse_log(text, status, return_code, error_hint):
    lines = _split_lines(_clean_text(text))
    hint = _clean_message(error_hint)
    result = dict.fromkeys(RESULT_KEYS)
    result["log_tail"] = _excerpt_tail(lines, LOG_TAIL_MAX_LINES)

    starts_idx, starts_cell = [], []
    error_in_testing, error_headers = [], []
    dead_kernel_idx = kernel_died_idx = timeout_idx = None
    for i, line in enumerate(lines):
        if "Start executing cell" in line:
            match = _CELL_START_RE.search(line[:_MATCH_MAX_CHARS])
            if match:
                starts_idx.append(i)
                starts_cell.append(int(match.group(1)))
        elif _ERROR_IN_TESTING in line:
            error_in_testing.append(i)
        elif _CELL_ERROR_HEADER in line:
            error_headers.append(i)
        if "DeadKernelError" in line:
            dead_kernel_idx = kernel_died_idx = i
        elif "Kernel died" in line:
            kernel_died_idx = i
        if "imeout reached (" in line and _TIMEOUT_LINE_RE.search(line[:_MATCH_MAX_CHARS]):
            timeout_idx = i

    def cell_index_before(pos):
        k = bisect.bisect_left(starts_idx, pos)
        return starts_cell[k - 1] if k > 0 else None

    if status == Status.TIMEOUT:
        result["category"] = FailureCategory.TIMEOUT
        result["cell_index"] = cell_index_before(len(lines))
        if timeout_idx is not None:
            result["exception_message"] = _clean_message(_LOG_PREFIX_RE.sub("", lines[timeout_idx].strip()))
        else:
            result["exception_message"] = hint
        return result

    if status == Status.ERROR:
        result["category"] = FailureCategory.HARNESS_ERROR
        if hint:
            result["exception_type"], result["exception_message"] = _split_hint(hint)
        return result

    cell_error = None
    if error_in_testing or error_headers:
        cell_error = _parse_cell_error(lines, error_in_testing, error_headers)

    # A "Kernel died" text inside a cell error block belongs to that cell's output, not to treon.
    if kernel_died_idx is not None and (cell_error is None or kernel_died_idx >= cell_error.end):
        result["category"] = FailureCategory.KERNEL_DIED
        result["cell_index"] = cell_index_before(len(lines))
        result["exception_type"] = "DeadKernelError"
        message = None
        if dead_kernel_idx is not None:
            match = _DEAD_KERNEL_RE.search(lines[dead_kernel_idx][:_MATCH_MAX_CHARS])
            if match:
                message = _clean_message(match.group(1))
        result["exception_message"] = message or "Kernel died"
        traceback = _last_python_traceback(lines)
        if traceback is not None:
            result["traceback_excerpt"] = _excerpt_tail(traceback[0], TRACEBACK_MAX_LINES)
        return result

    if cell_error is not None:
        result["category"] = FailureCategory.CELL_ERROR
        result["cell_index"] = cell_index_before(cell_error.start)
        result["cell_source_excerpt"] = _excerpt_head(cell_error.source_lines, CELL_SOURCE_MAX_LINES)
        result["traceback_excerpt"] = _excerpt_tail(cell_error.traceback_lines, TRACEBACK_MAX_LINES)
        if cell_error.exception is not None:
            result["exception_type"], result["exception_message"] = cell_error.exception
        elif hint:
            result["exception_message"] = hint
        return result

    result["category"] = FailureCategory.UNKNOWN
    result["cell_index"] = cell_index_before(len(lines))
    exception = None
    traceback = _last_python_traceback(lines)
    if traceback is not None:
        result["traceback_excerpt"] = _excerpt_tail(traceback[0], TRACEBACK_MAX_LINES)
        exception = traceback[1]
    if exception is None:
        exception = _generic_exception(lines)
    if exception is not None:
        result["exception_type"], result["exception_message"] = exception
    elif hint:
        result["exception_message"] = hint
    elif isinstance(return_code, int) and not isinstance(return_code, bool):
        result["exception_message"] = f"Process exited with return code {return_code}"
    return result


def _fallback_result(text, status, error_hint):
    result = dict.fromkeys(RESULT_KEYS)
    if status == Status.TIMEOUT:
        result["category"] = FailureCategory.TIMEOUT
    elif status == Status.ERROR:
        result["category"] = FailureCategory.HARNESS_ERROR
    else:
        result["category"] = FailureCategory.UNKNOWN
    try:
        result["exception_message"] = _clean_message(error_hint)
    except Exception:
        pass
    try:
        result["log_tail"] = _excerpt_tail(_split_lines(_clean_text(text)), LOG_TAIL_MAX_LINES)
    except Exception:
        pass
    return result


def _parse_cell_error(lines, error_in_testing, error_headers):
    """Locate nbclient's CellExecutionError text: cell source, optional stream sections, IPython traceback."""
    n = len(lines)
    testing_idx = error_in_testing[-1] if error_in_testing else None
    if testing_idx is not None:
        header_idx = next((i for i in error_headers if i > testing_idx), None)
        start = testing_idx
    else:
        header_idx = error_headers[-1]
        start = header_idx

    pos = (header_idx if header_idx is not None else start) + 1
    source_lines = None
    if header_idx is not None:
        j = pos
        while j < n and not lines[j].strip():
            j += 1
        if j < n and lines[j].strip() == _SEPARATOR:
            close = next((k for k in range(j + 1, n) if lines[k].strip() == _SEPARATOR), None)
            if close is None:
                source_lines, pos = lines[j + 1 :], n
            else:
                source_lines, pos = lines[j + 1 : close], close + 1

    end = max(_find_block_end(lines, pos), pos)
    tb_start = pos
    while tb_start < end and not lines[tb_start].strip():
        tb_start += 1
    if tb_start < end and _STREAM_HEADER_RE.fullmatch(lines[tb_start].strip()):
        tb_start = _skip_stream_sections(lines, tb_start, end)

    traceback_lines = _trim_blank(lines[tb_start:end])
    exception = _find_exception(traceback_lines)
    return _CellError(start, end, source_lines, traceback_lines, exception)


def _skip_stream_sections(lines, header_idx, end):
    """Return the index right after the "------------------" closing the stream sections.

    Stream text may itself contain a separator line, so prefer a separator followed by the start of an IPython
    traceback; otherwise the last separator wins (e.g. a single-line "UsageError: ..." traceback follows it).
    """
    separators = [i for i in range(header_idx + 1, end) if lines[i].strip() == _SEPARATOR]
    for i in separators:
        nxt = i + 1
        while nxt < end and not lines[nxt].strip():
            nxt += 1
        if nxt < end and _looks_like_traceback_start(lines[nxt]):
            return i + 1
    if separators:
        return separators[-1] + 1
    for i in range(header_idx + 1, end):
        if _looks_like_traceback_start(lines[i]):
            return i
    return end


def _looks_like_traceback_start(line):
    stripped = line.strip()[:_MATCH_MAX_CHARS]
    if _LONG_DASH_RE.fullmatch(stripped) or _TB_HEADER in stripped:
        return True
    return stripped.startswith(("Cell In[", "Input In [", "File ", "<ipython-input-"))


def _find_block_end(lines, start):
    for i in range(start, len(lines)):
        stripped = lines[i].strip()
        if stripped == "TEST RESULT" and i > start and _LONG_DASH_RE.fullmatch(lines[i - 1].strip()):
            return i - 1
        if _NOISE_RE.match(stripped[:_MATCH_MAX_CHARS]):
            return i
    return len(lines)


def _find_exception(traceback_lines):
    """(type, message) from the last exception line of an IPython traceback; the message may span lines."""
    found_idx = weak_idx = None
    found = weak = None
    for i in range(len(traceback_lines) - 1, -1, -1):
        exception = _parse_exception_line(traceback_lines[i])
        if exception is None:
            continue
        if exception.strong:
            found_idx, found = i, exception
            break
        if weak is None:
            weak_idx, weak = i, exception
    if found is None:
        found_idx, found = weak_idx, weak
    if found is None:
        return None

    message_lines = [found.message or ""]
    size = len(message_lines[0])
    for line in traceback_lines[found_idx + 1 :]:
        if size > MESSAGE_MAX_CHARS:
            break
        message_lines.append(line)
        size += len(line) + 1
    return found.exc_type, _clean_message("\n".join(message_lines))


def _last_python_traceback(lines):
    """(block lines, (type, message) or None) of the last plain Python traceback, or None."""
    header = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].lstrip().startswith(_TB_HEADER):
            header = i
            break
    if header is None:
        return None

    n = len(lines)
    i = header + 1
    while i < n and lines[i][:1] in (" ", "\t"):
        i += 1
    exception = None
    if i < n and lines[i].strip():
        parsed = _parse_exception_line(lines[i])
        if parsed is not None:
            exception = (parsed.exc_type, _clean_message(parsed.message))
            i += 1
    return lines[header:i], exception


def _generic_exception(lines):
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i][:_MATCH_MAX_CHARS]
        if ":" not in line or not line.split(":", 1)[0].rstrip().endswith(_GENERIC_EXC_SUFFIXES) or _ALL_CAPS_TOKEN_RE.match(line):
            continue
        match = _GENERIC_EXC_RE.match(line)
        if match:
            return _clean_type(match.group(1)), _clean_message(match.group(2))
    return None


def _parse_exception_line(line):
    stripped = line.strip()[:_MATCH_MAX_CHARS]
    match = _EXC_LINE_RE.fullmatch(stripped)
    if not match:
        return None
    exc_type = _clean_type(match.group(1))
    if not exc_type or not (exc_type[0].isalpha() or exc_type[0] == "_"):
        return None
    message = match.group(2)
    strong = exc_type in _KNOWN_EXC_NAMES or bool(_EXC_SUFFIX_RE.fullmatch(exc_type))
    # Weak candidates (e.g. custom exception classes) must look like "CamelName: message".
    if not strong and (message is None or not exc_type[0].isupper()):
        return None
    return _Exc(exc_type, message, strong)


def _split_hint(hint):
    match = _HINT_RE.match(hint)
    if match:
        exc_type = _clean_type(match.group(1))
        if exc_type and (exc_type in _KNOWN_EXC_NAMES or _EXC_SUFFIX_RE.fullmatch(exc_type)):
            return exc_type, _clean_message(match.group(2))
    return None, hint


def _last_component(name):
    return name.rsplit(".", 1)[-1]


def _clean_type(name):
    return _last_component(name.strip())[:_TYPE_MAX_CHARS] or None


def _to_str(value):
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8", "replace")
    return str(value)


def _clean_text(text):
    """ANSI-free text with "\\n" line endings, without control characters and carriage-return overwrites."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = _to_str(text)
    # Lone surrogates (e.g. from surrogateescape decoding) cannot be encoded later on.
    text = text.encode("utf-8", "replace").decode("utf-8", "replace")
    text = strip_ansi(text).replace("\r\n", "\n")
    text = _CONTROL_RE.sub("", text)
    if "\r" in text:
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if "\r" in line:
                # Progress bars redraw the line with "\r": keep what a terminal would finally show.
                parts = [part for part in line.split("\r") if part]
                lines[i] = parts[-1] if parts else ""
        text = "\n".join(lines)
    return text


def _split_lines(text):
    if not text:
        return []
    return [line.rstrip() for line in text.split("\n")]


def _clean_message(message):
    if message is None:
        return None
    lines = _split_lines(_clean_text(message))
    text = "\n".join(_trim_blank(lines)).strip()
    return text[:MESSAGE_MAX_CHARS].rstrip() or None


def _trim_blank(lines):
    if not lines:
        return []
    start, end = 0, len(lines)
    while end > start and not lines[end - 1].strip():
        end -= 1
    while start < end and not lines[start].strip():
        start += 1
    return lines[start:end]


def _excerpt_head(lines, max_lines):
    if not lines:
        return None
    start = 0
    while start < len(lines) and not lines[start].strip():
        start += 1
    text = "\n".join(lines[start : start + max_lines]).rstrip()
    encoded = text.encode("utf-8", "replace")
    if len(encoded) > EXCERPT_MAX_BYTES:
        text = encoded[:EXCERPT_MAX_BYTES].decode("utf-8", "ignore").rstrip()
    return text or None


def _excerpt_tail(lines, max_lines):
    if not lines:
        return None
    end = len(lines)
    while end > 0 and not lines[end - 1].strip():
        end -= 1
    text = "\n".join(_trim_blank(lines[max(0, end - max_lines) : end])).rstrip()
    encoded = text.encode("utf-8", "replace")
    if len(encoded) > EXCERPT_MAX_BYTES:
        text = encoded[-EXCERPT_MAX_BYTES:].decode("utf-8", "ignore").rstrip()
    return text or None

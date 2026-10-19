import hashlib
import random
import re
import time
from pathlib import Path

import pytest

from nightly_report.constants import (
    CELL_SOURCE_MAX_LINES,
    EXCERPT_MAX_BYTES,
    LOG_TAIL_MAX_LINES,
    MESSAGE_MAX_CHARS,
    TRACEBACK_MAX_LINES,
    ErrorClass,
    FailureCategory,
    Status,
)
from nightly_report.error_parser import analyze_failure, classify, compute_signature, normalize_message, parse_log, strip_ansi

LOGS_DIR = Path(__file__).resolve().parent / "fixtures" / "logs"

RESULT_KEYS = [
    "category",
    "cell_index",
    "cell_source_excerpt",
    "exception_type",
    "exception_message",
    "traceback_excerpt",
    "log_tail",
]
SIGNATURE_RE = re.compile(r"^[0-9a-f]{12}$")
SEPARATOR = "-" * 18
RED = "\x1b[0;31m"
RESET = "\x1b[0m"


def read_log(name):
    return (LOGS_DIR / name).read_bytes().decode("utf-8")


def ipython_traceback(ename, evalue, frames=1):
    lines = [f"{RED}{'-' * 75}{RESET}", f"{RED}{ename}{RESET}{' ' * 20}Traceback (most recent call last)"]
    for i in range(frames):
        lines += [f"Cell \x1b[0;32mIn[{i + 1}], line 1\x1b[0m", f"\x1b[0;32m----> 1\x1b[0m step_{i}()", ""]
    lines.append(f"{RED}{ename}{RESET}: {evalue}")
    return lines


def cell_error_log(source, traceback, streams=None, before=(), after=(), notebook="test_x.ipynb"):
    lines = list(before) + [f"ERROR in testing {notebook}", "", "An error occurred while executing the following cell:", SEPARATOR]
    lines += list(source) + [SEPARATOR, ""]
    if streams:
        lines.append("")
        for name, text in streams:
            lines += [f"----- {name} -----"] + list(text)
        lines.append(SEPARATOR)
    else:
        lines.append("")
    lines += [""] + list(traceback) + ["", "", ""]
    lines += ["", "-" * 71, "TEST RESULT", "-" * 71, f"{notebook}       -- FAILED ", "-" * 71]
    lines += ["0 succeeded, 1 failed, out of 1 notebooks tested.", "-" * 71, ""]
    lines += list(after)
    return "\n".join(lines) + "\n"


def assert_well_formed(result):
    assert list(result) == RESULT_KEYS
    assert result["category"] in FailureCategory.ALL
    assert result["cell_index"] is None or isinstance(result["cell_index"], int)
    for key in RESULT_KEYS[2:]:
        value = result[key]
        if value is None:
            continue
        assert isinstance(value, str) and value
        assert "\x1b" not in value and "\r" not in value
        assert value == value.rstrip()
        assert all(line == line.rstrip() for line in value.split("\n"))
    for key in ("cell_source_excerpt", "traceback_excerpt", "log_tail"):
        if result[key] is not None:
            assert len(result[key].encode("utf-8")) <= EXCERPT_MAX_BYTES
    if result["cell_source_excerpt"] is not None:
        assert len(result["cell_source_excerpt"].split("\n")) <= CELL_SOURCE_MAX_LINES
    if result["traceback_excerpt"] is not None:
        assert len(result["traceback_excerpt"].split("\n")) <= TRACEBACK_MAX_LINES
    if result["log_tail"] is not None:
        assert len(result["log_tail"].split("\n")) <= LOG_TAIL_MAX_LINES
    if result["exception_message"] is not None:
        assert len(result["exception_message"]) <= MESSAGE_MAX_CHARS


FIXTURES = {
    "cell_error.log": dict(
        status=Status.FAILED,
        category=FailureCategory.CELL_ERROR,
        exception_type="ValueError",
        exception_message="Expected a positive value, got -1",
        cell_index=4,
        error_class=ErrorClass.OTHER,
    ),
    "cell_error_with_stdout.log": dict(
        status=Status.FAILED,
        category=FailureCategory.CELL_ERROR,
        exception_type="KeyError",
        exception_message="'logits'",
        cell_index=7,
        error_class=ErrorClass.OTHER,
    ),
    "kernel_died.log": dict(
        status=Status.FAILED,
        category=FailureCategory.KERNEL_DIED,
        exception_type="DeadKernelError",
        exception_message="Kernel died",
        cell_index=5,
        error_class=ErrorClass.CRASH,
    ),
    "oom_kernel_died.log": dict(
        status=Status.FAILED,
        category=FailureCategory.KERNEL_DIED,
        exception_type="DeadKernelError",
        exception_message="Kernel died",
        cell_index=9,
        error_class=ErrorClass.MEMORY,
    ),
    "timeout.log": dict(
        status=Status.TIMEOUT,
        category=FailureCategory.TIMEOUT,
        exception_type=None,
        exception_message="Timeout reached (7200s), process killed",
        cell_index=12,
        error_class=ErrorClass.TIMEOUT,
    ),
    "module_not_found.log": dict(
        status=Status.FAILED,
        category=FailureCategory.CELL_ERROR,
        exception_type="ModuleNotFoundError",
        exception_message="No module named 'openvino_genai'",
        cell_index=1,
        error_class=ErrorClass.DEPENDENCY,
    ),
    "network_hf.log": dict(
        status=Status.FAILED,
        category=FailureCategory.CELL_ERROR,
        exception_type="ConnectionError",
        exception_message=None,
        cell_index=3,
        error_class=ErrorClass.NETWORK,
    ),
    "openvino_runtime.log": dict(
        status=Status.FAILED,
        category=FailureCategory.CELL_ERROR,
        exception_type="RuntimeError",
        exception_message=None,
        cell_index=6,
        error_class=ErrorClass.OPENVINO,
    ),
    "unknown.log": dict(
        status=Status.FAILED,
        category=FailureCategory.UNKNOWN,
        exception_type="AssertionError",
        exception_message="0.42 not greater than or equal to 0.5 : accuracy too low",
        cell_index=4,
        error_class=ErrorClass.ASSERTION,
    ),
}


def test_all_fixture_logs_are_covered():
    assert sorted(p.name for p in LOGS_DIR.glob("*.log")) == sorted(FIXTURES)


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_fixture_logs(name):
    expected = FIXTURES[name]
    result = analyze_failure(read_log(name), expected["status"], return_code=1)

    assert list(result) == RESULT_KEYS + ["error_class", "signature"]
    parsed = {key: result[key] for key in RESULT_KEYS}
    assert_well_formed(parsed)
    assert parsed == parse_log(read_log(name), expected["status"], return_code=1)
    assert result["category"] == expected["category"]
    assert result["exception_type"] == expected["exception_type"]
    if expected["exception_message"] is not None:
        assert result["exception_message"] == expected["exception_message"]
    assert result["cell_index"] == expected["cell_index"]
    assert result["error_class"] == expected["error_class"]
    assert SIGNATURE_RE.match(result["signature"])
    assert result["signature"] == compute_signature(result["category"], result["exception_type"], result["exception_message"])
    assert result["log_tail"]


def test_cell_error_fixture_excerpts():
    result = parse_log(read_log("cell_error.log"), Status.FAILED)

    assert result["cell_source_excerpt"] == (
        "def check_positive(value):\n"
        "    if value < 0:\n"
        '        raise ValueError(f"Expected a positive value, got {value}")\n'
        "    return value\n"
        "\n"
        "check_positive(-1)"
    )
    traceback = result["traceback_excerpt"].split("\n")
    assert traceback[0] == "-" * 75
    assert traceback[1].startswith("ValueError") and traceback[1].endswith("Traceback (most recent call last)")
    assert "Cell In[3], line 6" in traceback
    assert '----> 3         raise ValueError(f"Expected a positive value, got {value}")' in traceback
    assert traceback[-1] == "ValueError: Expected a positive value, got -1"
    for noise in ("TEST RESULT", "-- FAILED", "OpenVINO after", "Package(s) not found", "ERROR in testing", "check_positive(value):\n    if"):
        assert noise not in result["traceback_excerpt"]
    assert result["log_tail"].split("\n")[-1] == "openvino_genai is missing in validation environment."
    assert "WARNING: Package(s) not found: openvino_genai" in result["log_tail"]


def test_stream_sections_are_not_part_of_the_traceback():
    result = parse_log(read_log("cell_error_with_stdout.log"), Status.FAILED)

    traceback = result["traceback_excerpt"]
    assert traceback.split("\n")[0] == "-" * 75
    assert traceback.split("\n")[-1] == "KeyError: 'logits'"
    for stream_text in ("Loading model...", "only printed output", "UserWarning", "warnings.warn", "----- stdout -----"):
        assert stream_text not in traceback
    assert result["exception_type"] == "KeyError"
    assert result["cell_source_excerpt"].split("\n")[-1] == 'logits = results["logits"]'


def test_kernel_died_fixture_uses_last_python_traceback():
    result = parse_log(read_log("kernel_died.log"), Status.FAILED)

    traceback = result["traceback_excerpt"].split("\n")
    assert traceback[0] == "Traceback (most recent call last):"
    assert traceback[-1] == "nbclient.exceptions.DeadKernelError: Kernel died"
    assert result["cell_source_excerpt"] is None
    assert "OpenVINO after notebook execution" not in result["traceback_excerpt"]


def test_timeout_fixture_normalizes_crlf_and_progress_bars():
    result = parse_log(read_log("timeout.log"), Status.TIMEOUT, return_code=-42)

    assert result["traceback_excerpt"] is None
    assert result["cell_source_excerpt"] is None
    tail = result["log_tail"].split("\n")
    assert tail[-1] == "[validate_notebooks] Timeout reached (7200s), process killed"
    assert "INFO:nncf:Statistics collection: 100%|##########| 300/300" in tail
    progress = [line for line in tail if line.startswith("model.safetensors")]
    assert progress == ["model.safetensors: 100%|##########| 1500M/1.50G [00:50<01:00, 30.0MB/s]"]


def test_openvino_fixture_keeps_multiline_message():
    result = parse_log(read_log("openvino_runtime.log"), Status.FAILED)

    message = result["exception_message"]
    assert message.startswith("Exception from src/inference/src/cpp/core.cpp:112:\n")
    assert "[ GENERAL_ERROR ] Check 'false' failed at" in message
    assert message.endswith("[GPU] Unsupported layout for reorder: bfyx f16 -> b_fs_yx_fsv16 u8")
    assert result["traceback_excerpt"].endswith(message)


def test_network_fixture_signature_ignores_addresses_and_request_ids():
    text = read_log("network_hf.log")
    first = analyze_failure(text, Status.FAILED)
    other = text.replace("0x7f3c1a2b4d90", "0x55d0c0ffee01").replace("6f1c2d3e-4b5a-6978-8a9b-0c1d2e3f4a5b", "0a1b2c3d-1111-2222-3333-444455556666")
    second = analyze_failure(other, Status.FAILED)

    assert first["exception_message"].startswith("(MaxRetryError(\"HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded")
    assert first["exception_message"] != second["exception_message"]
    assert first["signature"] == second["signature"]


def test_unknown_fixture_traceback_block():
    result = parse_log(read_log("unknown.log"), Status.FAILED)

    traceback = result["traceback_excerpt"].split("\n")
    assert traceback[0] == "Traceback (most recent call last):"
    assert traceback[-1] == "AssertionError: 0.42 not greater than or equal to 0.5 : accuracy too low"
    assert len(traceback) == 4


def test_ansi_codes_are_stripped_everywhere():
    source = ["\x1b[38;5;28;01mraise\x1b[39;00m ValueError()"]
    traceback = ipython_traceback("ValueError", "\x1b[1mbold\x1b[0m message")
    text = cell_error_log(source, traceback, before=["\x1b[32mStart executing cell 2\x1b[0m"])
    result = parse_log(text, Status.FAILED)

    assert_well_formed(result)
    assert result["cell_index"] == 2
    assert result["cell_source_excerpt"] == "raise ValueError()"
    assert result["exception_message"] == "bold message"


def test_cell_index_is_last_cell_started_before_the_error_block():
    text = cell_error_log(
        ["boom()"],
        ipython_traceback("NameError", "name 'boom' is not defined"),
        before=["[treon] Start executing cell 1", "Start executing cell 3", "Start executing cell 8"],
        after=["Start executing cell 99"],
    )
    assert parse_log(text, Status.FAILED)["cell_index"] == 8


def test_cell_index_is_none_without_cell_markers():
    result = parse_log(cell_error_log(["boom()"], ipython_traceback("NameError", "x")), Status.FAILED)
    assert result["cell_index"] is None
    assert result["exception_type"] == "NameError"


def test_cell_error_header_without_error_in_testing_line():
    text = "\n".join(
        ["Start executing cell 5", "An error occurred while executing the following cell:", SEPARATOR, "x = 1 / 0", SEPARATOR, "", "", ""]
        + ipython_traceback("ZeroDivisionError", "division by zero")
    )
    result = parse_log(text, Status.FAILED)
    assert result["category"] == FailureCategory.CELL_ERROR
    assert result["cell_index"] == 5
    assert result["cell_source_excerpt"] == "x = 1 / 0"
    assert result["exception_type"] == "ZeroDivisionError"
    assert result["exception_message"] == "division by zero"


def test_error_in_testing_without_cell_source():
    text = "\n".join(["Start executing cell 2", "ERROR in testing test_x.ipynb", ""] + ipython_traceback("TypeError", "bad operand"))
    result = parse_log(text, Status.FAILED)
    assert result["category"] == FailureCategory.CELL_ERROR
    assert result["cell_source_excerpt"] is None
    assert result["exception_type"] == "TypeError"
    assert result["traceback_excerpt"].endswith("TypeError: bad operand")


def test_chained_exceptions_report_the_last_one():
    traceback = ipython_traceback("KeyError", "'a'") + ["", "During handling of the above exception, another exception occurred:", ""]
    traceback += ipython_traceback("ValueError", "converted")
    result = parse_log(cell_error_log(["f()"], traceback), Status.FAILED)
    assert (result["exception_type"], result["exception_message"]) == ("ValueError", "converted")


@pytest.mark.parametrize(
    "last_line, expected_type, expected_message",
    [
        (f"{RED}KeyboardInterrupt{RESET}: ", "KeyboardInterrupt", None),
        ("KeyboardInterrupt", "KeyboardInterrupt", None),
        (f"{RED}MyCustomProblem{RESET}: something odd", "MyCustomProblem", "something odd"),
        (f"{RED}torch.OutOfMemoryError{RESET}: CUDA out of memory", "OutOfMemoryError", "CUDA out of memory"),
        (f"{RED}Exception{RESET}: plain", "Exception", "plain"),
    ],
)
def test_exception_line_variants(last_line, expected_type, expected_message):
    traceback = ipython_traceback("X", "y")[:-1] + [last_line]
    result = parse_log(cell_error_log(["f()"], traceback), Status.FAILED)
    assert result["exception_type"] == expected_type
    assert result["exception_message"] == expected_message


def test_syntax_error_after_stream_section():
    traceback = ["  Cell \x1b[0;32mIn[2], line 1\x1b[0m", "    x = (", "        ^", f"{RED}SyntaxError{RESET}: '(' was never closed"]
    streams = [("stdout", ["------------------", "SyntaxWarning: printed text"])]
    result = parse_log(cell_error_log(["x = ("], traceback, streams=streams), Status.FAILED)
    assert result["exception_type"] == "SyntaxError"
    assert result["traceback_excerpt"].split("\n")[0] == "  Cell In[2], line 1"
    assert "printed text" not in result["traceback_excerpt"]


def test_single_line_traceback_after_stream_section():
    streams = [("stderr", ["some warning"])]
    result = parse_log(cell_error_log(["%foo"], [f"{RED}UsageError{RESET}: Line magic function `%foo` not found."], streams=streams), Status.FAILED)
    assert result["exception_type"] == "UsageError"
    assert result["traceback_excerpt"] == "UsageError: Line magic function `%foo` not found."


def test_kernel_died_text_inside_cell_output_is_a_cell_error():
    streams = [("stdout", ["Kernel died while waiting for execute reply."])]
    text = cell_error_log(["run_child_notebook()"], ipython_traceback("RuntimeError", "child failed"), streams=streams)
    result = parse_log(text, Status.FAILED)
    assert result["category"] == FailureCategory.CELL_ERROR
    assert result["exception_type"] == "RuntimeError"


def test_kernel_died_without_traceback():
    result = parse_log("Start executing cell 3\nKernel died while waiting for execute reply.\n", Status.FAILED)
    assert result["category"] == FailureCategory.KERNEL_DIED
    assert (result["exception_type"], result["exception_message"]) == ("DeadKernelError", "Kernel died")
    assert result["cell_index"] == 3
    assert result["traceback_excerpt"] is None

    truncated = parse_log('Traceback (most recent call last):\n  File "client.py", line 1\n    raise DeadKernelError("Kernel died")', Status.FAILED)
    assert truncated["category"] == FailureCategory.KERNEL_DIED
    assert truncated["exception_message"] == "Kernel died"
    assert truncated["traceback_excerpt"].startswith("Traceback (most recent call last):")


def test_timeout_keeps_cell_index_and_ignores_errors():
    text = "Start executing cell 1\nStart executing cell 4\nValueError: not relevant\nTraceback (most recent call last):\n  File x\nOSError: x\n"
    result = parse_log(text, Status.TIMEOUT)
    assert result["category"] == FailureCategory.TIMEOUT
    assert result["cell_index"] == 4
    assert result["exception_type"] is None
    assert result["exception_message"] is None
    assert result["traceback_excerpt"] is None


def test_timeout_message_of_legacy_harness_line_and_hint():
    text = "Start executing cell 2\n\nNotebook test [test_x.ipynb] timeout reached (60s), killing process...\n"
    assert parse_log(text, Status.TIMEOUT)["exception_message"] == "Notebook test [test_x.ipynb] timeout reached (60s), killing process..."
    assert parse_log("Start executing cell 2", Status.TIMEOUT, error_hint="Timed out")["exception_message"] == "Timed out"


@pytest.mark.parametrize(
    "hint, expected_type, expected_message",
    [
        ('FileNotFoundError: Patched notebook "test_x.ipynb" does not exist.', "FileNotFoundError", 'Patched notebook "test_x.ipynb" does not exist.'),
        ("subprocess.CalledProcessError: pip failed", "CalledProcessError", "pip failed"),
        ("Job failed before the notebook ran", None, "Job failed before the notebook ran"),
        ("Note: not an exception", None, "Note: not an exception"),
        ("\x1b[31mRuntimeError\x1b[0m: colored", "RuntimeError", "colored"),
        (None, None, None),
        ("", None, None),
    ],
)
def test_harness_error(hint, expected_type, expected_message):
    result = analyze_failure("Start executing cell 3\nTraceback (most recent call last):\n  File x\nValueError: y\n", Status.ERROR, error_hint=hint)
    assert result["category"] == FailureCategory.HARNESS_ERROR
    assert result["exception_type"] == expected_type
    assert result["exception_message"] == expected_message
    assert result["cell_index"] is None
    assert result["error_class"] == ErrorClass.HARNESS


def test_unknown_generic_fallback_ignores_all_caps_noise():
    text = "\n".join(
        [
            "Installing requirements",
            "subprocess.CalledProcessError: Command '['pip', 'install', 'foo']' returned non-zero exit status 1.",
            "ERROR: Exception: something pip printed",
            "WARNING: Package(s) not found: openvino_genai",
            "OpenVINO after notebook execution: 2026.4.1",
        ]
    )
    result = analyze_failure(text, Status.FAILED)
    assert result["category"] == FailureCategory.UNKNOWN
    assert result["exception_type"] == "CalledProcessError"
    assert result["exception_message"].startswith("Command '['pip', 'install', 'foo']'")
    assert result["traceback_excerpt"] is None
    assert result["error_class"] == ErrorClass.DEPENDENCY


def test_unknown_without_any_exception():
    text = "WARNING: Package(s) not found: openvino_genai\nOpenVINO after notebook execution: 2026.4.1\n"
    result = parse_log(text, Status.FAILED, return_code=-9)
    assert result["category"] == FailureCategory.UNKNOWN
    assert result["exception_type"] is None
    assert result["exception_message"] == "Process exited with return code -9"
    assert parse_log(text, Status.FAILED, error_hint="treon crashed")["exception_message"] == "treon crashed"
    assert parse_log(text, Status.FAILED)["exception_message"] is None


GARBAGE_INPUTS = [
    None,
    "",
    " \n\n\t\n",
    "\x00\x01\x02\x7fgarbage\x1b[",
    "\x1b]0;title\x07\x1b[2J\x1b[1;1H\x1b(B",
    "\ud800 lone surrogate \udcff",
    "Start executing cell 99999999999999999999999\nStart executing cell -1",
    "ERROR in testing\nAn error occurred while executing the following cell:\n" + SEPARATOR,
    "An error occurred while executing the following cell:\n" + SEPARATOR + "\nsource without end",
    "----- stdout -----\n" * 50,
    "Traceback (most recent call last):",
    "Traceback (most recent call last):\n  File x\n",
    "DeadKernelError",
    ": : : :\nError:\n.Error: x\n_: y",
    b"bytes \xff\xfe log\nValueError: from bytes",
    42,
]


@pytest.mark.parametrize("status", list(Status.FAILING) + ["passed", None, "bogus"])
@pytest.mark.parametrize("text", GARBAGE_INPUTS, ids=range(len(GARBAGE_INPUTS)))
def test_garbage_input_never_raises(text, status):
    result = analyze_failure(text, status, return_code=None, error_hint=None)
    assert_well_formed({key: result[key] for key in RESULT_KEYS})
    assert result["error_class"] in ErrorClass.ALL
    assert SIGNATURE_RE.match(result["signature"])


def test_random_binary_input_never_raises():
    rng = random.Random(0)
    data = bytes(rng.randrange(256) for _ in range(200_000))
    for text in (data.decode("latin-1"), data.decode("utf-8", "surrogateescape"), data.decode("utf-8", "replace")):
        for status in Status.FAILING:
            result = analyze_failure(text, status, return_code=1, error_hint="hint")
            assert_well_formed({key: result[key] for key in RESULT_KEYS})


def test_empty_input():
    result = parse_log(None, Status.FAILED)
    assert result == dict.fromkeys(RESULT_KEYS) | {"category": FailureCategory.UNKNOWN}
    assert parse_log("", Status.TIMEOUT)["category"] == FailureCategory.TIMEOUT
    assert parse_log("", Status.ERROR)["category"] == FailureCategory.HARNESS_ERROR


def test_huge_log_respects_caps():
    filler = "".join(f"Start executing cell {i % 50}\nprogress line {i} " + "ж" * 80 + "\n" for i in range(30_000))
    source = [f"line_{i} = {i}  # " + "x" * 300 for i in range(500)]
    traceback = ipython_traceback("RecursionError", "maximum recursion depth exceeded " + "y" * 2_000, frames=2_000)
    text = filler + cell_error_log(source, traceback, after=["z" * 100_000] * 30)
    assert len(text.encode("utf-8")) > 5 * 1024 * 1024

    started = time.perf_counter()
    result = analyze_failure(text, Status.FAILED)
    assert time.perf_counter() - started < 20

    assert_well_formed({key: result[key] for key in RESULT_KEYS})
    assert result["category"] == FailureCategory.CELL_ERROR
    assert result["exception_type"] == "RecursionError"
    assert result["cell_index"] == 29_999 % 50
    assert result["cell_source_excerpt"].startswith("line_0 = 0")
    assert result["traceback_excerpt"].split("\n")[-1].startswith("RecursionError: maximum recursion depth exceeded y")
    assert len(result["exception_message"]) == MESSAGE_MAX_CHARS


def test_huge_last_traceback_line_keeps_the_tail():
    traceback = ipython_traceback("ValueError", "z" * 50_000)
    result = parse_log(cell_error_log(["f()"], traceback), Status.FAILED)
    assert_well_formed(result)
    assert result["traceback_excerpt"].endswith("z" * 100)
    assert result["exception_message"] == "z" * MESSAGE_MAX_CHARS


def test_huge_single_line_input():
    text = "x" * (5 * 1024 * 1024) + "Error: " + "a" * 1000
    for status in Status.FAILING:
        result = parse_log(text, status)
        assert_well_formed(result)
        assert result["log_tail"].endswith("a" * 1000)


def test_multibyte_tail_cap_keeps_valid_utf8():
    text = "\n".join("€" * 500 for _ in range(200))
    tail = parse_log(text, Status.FAILED)["log_tail"]
    assert len(tail.encode("utf-8")) <= EXCERPT_MAX_BYTES
    assert set(tail) <= {"€", "\n"}


@pytest.mark.parametrize(
    "text, expected",
    [
        ("\x1b[0;31mValueError\x1b[0m: boom", "ValueError: boom"),
        ("\x1b[38;5;28;01mraise\x1b[39;00m", "raise"),
        ("\x1b]0;window title\x07visible", "visible"),
        ("\x1b[2K\x1b[1Gline", "line"),
        ("no escapes", "no escapes"),
        ("", ""),
        (None, ""),
    ],
)
def test_strip_ansi(text, expected):
    assert strip_ansi(text) == expected


@pytest.mark.parametrize(
    "message, expected",
    [
        (None, ""),
        ("", ""),
        ("No such file or directory: '/tmp/tmpab12cd/model.xml'", "No such file or directory: '<path>'"),
        ("File C:\\Users\\runner\\AppData\\Local\\Temp\\x.bin is locked", "File <path> is locked"),
        ("object at 0x7f3c1a2b4d90", "object at <hex>"),
        ("Request ID: 6f1c2d3e-4b5a-6978-8a9b-0c1d2e3f4a5b", "Request ID: <uuid>"),
        ("revision 3f2a9c1e5b7d4f60a1b2c3d4e5f60718293a4b5c not found", "revision <hex> not found"),
        ("Tried to allocate 1.50 GiB (GPU 0; 22.17 GiB total)", "Tried to allocate <n> GiB (GPU <n>; <n> GiB total)"),
        ("Exception from src/inference/src/cpp/core.cpp:112:", "Exception from src/inference/src/cpp/core.cpp:<n>:"),
        ("Model   Not\n\tFound", "Model Not Found"),
        ("\x1b[31mRed\x1b[0m", "Red"),
        ("see https://huggingface.co/docs", "see https://huggingface.co/docs"),
    ],
)
def test_normalize_message(message, expected):
    assert normalize_message(message) == expected


def test_normalize_message_truncates():
    assert len(normalize_message("word " * 1000)) == 300
    assert normalize_message("a" * 100_000) == "a" * 300


@pytest.mark.parametrize(
    "first, second",
    [
        ("Cannot open /tmp/tmpk2j3/model.xml", "Cannot open /home/runner/work/x/y/model.xml"),
        ("<Foo object at 0x7f3c1a2b4d90>", "<Foo object at 0x55d0c0ffee00>"),
        ("Process 12345 exited", "Process 678 exited"),
        ("Unable to allocate 3.5 GiB for an array with shape (1024, 2048)", "Unable to allocate 12.25 GiB for an array with shape (4096, 4096)"),
        ("error at line 12 col 3", "error   at line 7 col 40"),
        ("session 6f1c2d3e-4b5a-6978-8a9b-0c1d2e3f4a5b expired", "session 0a1b2c3d-1111-2222-3333-444455556666 expired"),
        ("D:\\a\\openvino_notebooks\\x.py failed", "C:\\Users\\x\\y.py failed"),
    ],
)
def test_signature_is_stable_across_run_specific_details(first, second):
    assert compute_signature("cell_error", "RuntimeError", first) == compute_signature("cell_error", "RuntimeError", second)


def test_signature_distinguishes_errors():
    base = compute_signature("cell_error", "ValueError", "boom")
    assert base != compute_signature("cell_error", "TypeError", "boom")
    assert base != compute_signature("unknown", "ValueError", "boom")
    assert base != compute_signature("cell_error", "ValueError", "bang")
    assert base != compute_signature("cell_error", None, "boom")


def test_signature_format_and_formula():
    for args in [("cell_error", "ValueError", "boom"), ("timeout", None, None), (None, None, None), ("unknown", "", "x" * 10_000)]:
        signature = compute_signature(*args)
        assert SIGNATURE_RE.match(signature)
    expected = hashlib.sha1("cell_error|ValueError|line <n>: <path>".encode()).hexdigest()[:12]
    assert compute_signature("cell_error", "ValueError", "line 3: /tmp/a") == expected


@pytest.mark.parametrize(
    "category, exception_type, message, text, expected",
    [
        # 1-2: category based
        ("timeout", "MemoryError", "out of memory", "", ErrorClass.TIMEOUT),
        ("harness_error", "ModuleNotFoundError", "No module named x", "", ErrorClass.HARNESS),
        # 3: disk
        ("cell_error", "OSError", "[Errno 28] No space left on device: '/tmp/x'", "", ErrorClass.DISK),
        ("cell_error", "MemoryError", "No space left on device", "", ErrorClass.DISK),
        ("unknown", None, None, "OSError: [Errno 28] No space left on device", ErrorClass.DISK),
        # 4: memory
        ("cell_error", "MemoryError", "", "", ErrorClass.MEMORY),
        ("cell_error", "torch.OutOfMemoryError", "", "", ErrorClass.MEMORY),
        ("cell_error", "RuntimeError", "CUDA out of memory. Tried to allocate 2.00 GiB", "", ErrorClass.MEMORY),
        ("cell_error", "RuntimeError", "std::bad_alloc", "", ErrorClass.MEMORY),
        ("cell_error", "OSError", "[Errno 12] Cannot allocate memory", "", ErrorClass.MEMORY),
        ("cell_error", "_ArrayMemoryError", "Unable to allocate 3.5 GiB for an array", "", ErrorClass.MEMORY),
        ("unknown", None, None, "kernel: oom-kill:constraint=CONSTRAINT_NONE", ErrorClass.MEMORY),
        ("kernel_died", "DeadKernelError", "Kernel died", "Runner OOM detected", ErrorClass.MEMORY),
        ("kernel_died", "DeadKernelError", "Kernel died", "terminate called after throwing an instance of 'std::bad_alloc'", ErrorClass.MEMORY),
        ("cell_error", "ValueError", "boom", "std::bad_alloc in log tail is ignored for cell errors", ErrorClass.OTHER),
        ("cell_error", "ValueError", "ZOOM level", "", ErrorClass.OTHER),
        # 5: crash
        ("kernel_died", "DeadKernelError", "Kernel died", "", ErrorClass.CRASH),
        ("kernel_died", "DeadKernelError", "Kernel died", "Read timed out", ErrorClass.CRASH),
        # 6: dependency
        ("cell_error", "ModuleNotFoundError", "No module named 'x'", "", ErrorClass.DEPENDENCY),
        ("cell_error", "ImportError", "cannot import name 'y'", "", ErrorClass.DEPENDENCY),
        ("cell_error", "importlib.metadata.PackageNotFoundError", "No package metadata was found for x", "", ErrorClass.DEPENDENCY),
        ("cell_error", "RuntimeError", "ERROR: No matching distribution found for foo==1.0", "", ErrorClass.DEPENDENCY),
        ("cell_error", "RuntimeError", "ResolutionImpossible: for help visit", "", ErrorClass.DEPENDENCY),
        ("cell_error", "RuntimeError", "Could not find a version that satisfies the requirement foo", "", ErrorClass.DEPENDENCY),
        ("cell_error", "RuntimeError", "error: subprocess-exited-with-error", "", ErrorClass.DEPENDENCY),
        ("cell_error", "OSError", "DLL load failed while importing _pyopenvino", "", ErrorClass.DEPENDENCY),
        ("cell_error", "CalledProcessError", "Command '['python', '-m', 'pip', 'install', 'x']' returned 1", "", ErrorClass.DEPENDENCY),
        ("cell_error", "CalledProcessError", "Command '['git', 'clone', 'x']' returned 128", "", ErrorClass.OTHER),
        ("unknown", None, None, "ERROR: No matching distribution found for foo", ErrorClass.DEPENDENCY),
        # 7: network
        *[
            ("cell_error", name, "", "", ErrorClass.NETWORK)
            for name in (
                "requests.exceptions.ConnectionError",
                "ConnectTimeout",
                "ReadTimeout",
                "ConnectTimeoutError",
                "ReadTimeoutError",
                "HTTPError",
                "urllib.error.URLError",
                "SSLError",
                "MaxRetryError",
                "ProxyError",
                "HfHubHTTPError",
                "LocalEntryNotFoundError",
                "ChunkedEncodingError",
                "IncompleteRead",
                "RemoteDisconnected",
                "socket.gaierror",
            )
        ],
        ("cell_error", "OSError", "HTTPSConnectionPool(host='x', port=443): Max retries exceeded with url", "", ErrorClass.NETWORK),
        ("cell_error", "OSError", "[Errno 104] Connection reset by peer", "", ErrorClass.NETWORK),
        ("cell_error", "OSError", "[Errno 111] Connection refused", "", ErrorClass.NETWORK),
        ("cell_error", "OSError", "('Connection aborted.', RemoteDisconnected('x'))", "", ErrorClass.NETWORK),
        ("cell_error", "OSError", "[Errno -3] Temporary failure in name resolution", "", ErrorClass.NETWORK),
        ("cell_error", "OSError", "[Errno -2] Name or service not known", "", ErrorClass.NETWORK),
        ("cell_error", "OSError", "Read timed out. (read timeout=10)", "", ErrorClass.NETWORK),
        ("cell_error", "RuntimeError", "Too Many Requests for url", "", ErrorClass.NETWORK),
        ("cell_error", "OSError", "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed", "", ErrorClass.NETWORK),
        ("cell_error", "OSError", "429 Client Error: Too Many Requests for url: https://huggingface.co/api", "", ErrorClass.NETWORK),
        ("cell_error", "OSError", "504 Server Error: Gateway Time-out for url: https://huggingface.co/api", "", ErrorClass.NETWORK),
        ("cell_error", "OSError", "HTTP Error 503: Service Unavailable", "", ErrorClass.NETWORK),
        ("cell_error", "RuntimeError", "request failed with status code 502", "", ErrorClass.NETWORK),
        ("unknown", None, None, "requests.exceptions.ReadTimeout: Read timed out.", ErrorClass.NETWORK),
        ("cell_error", "IndexError", "index 503 is out of bounds for axis 0 with size 503", "", ErrorClass.OTHER),
        ("cell_error", "AssertionError", "Read timed out", "", ErrorClass.NETWORK),
        # 8: openvino
        ("cell_error", "RuntimeError", "Exception from src/inference/src/cpp/core.cpp:112:", "", ErrorClass.OPENVINO),
        ("cell_error", "RuntimeError", "[ GENERAL_ERROR ] something failed", "", ErrorClass.OPENVINO),
        ("cell_error", "RuntimeError", "Check 'shape.rank().is_static()' failed at src/core/shape_inference/x.hpp:42:", "", ErrorClass.OPENVINO),
        ("unknown", None, None, "Exception from src/plugins/intel_cpu/src/node.cpp:1:", ErrorClass.OPENVINO),
        # 9: assertion
        ("cell_error", "AssertionError", "1 != 2", "", ErrorClass.ASSERTION),
        ("unknown", "AssertionError", None, "", ErrorClass.ASSERTION),
        # 10: other
        ("cell_error", "ValueError", "boom", "", ErrorClass.OTHER),
        ("cell_error", None, None, "", ErrorClass.OTHER),
        ("unknown", None, None, "", ErrorClass.OTHER),
    ],
)
def test_classify(category, exception_type, message, text, expected):
    assert classify(category, exception_type, message, text) == expected


def test_classify_never_raises():
    for args in [(None, None, None, None), ("cell_error", 123, object(), b"bytes"), ("unknown", "\x00", "\udcff", "\ud800")]:
        assert classify(*args) in ErrorClass.ALL
    assert classify("cell_error", "ValueError", "boom") == ErrorClass.OTHER

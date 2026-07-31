"""Wrapper around treon that prints the full source of a failing cell.

treon only logs the kernel traceback on failure, which makes it hard to tell
which cell failed from the CI logs. nbconvert's ``CellExecutionError`` already
carries the full source of the offending cell in its string form, so we
intercept it, print it, and let treon proceed as usual.
"""

import treon.task as treon_task
from nbconvert.preprocessors import CellExecutionError

_original_execute_notebook = treon_task.execute_notebook


def _execute_notebook_with_cell_dump(path, verbose=False):
    try:
        return _original_execute_notebook(path, verbose)
    except CellExecutionError as cell_error:
        print("\n" + "=" * 80, flush=True)
        print(f"FAILED CELL in {path}", flush=True)
        print("=" * 80, flush=True)
        # str(cell_error) includes the full source of the failing cell.
        print(str(cell_error), flush=True)
        print("=" * 80 + "\n", flush=True)
        raise


treon_task.execute_notebook = _execute_notebook_with_cell_dump

from treon.treon import main  # noqa: E402  (import after monkeypatch is intentional)

main()

import sys, os
import subprocess  # nosec - disable B404:import-subprocess check
from pathlib import Path

spellcheck_dir = Path(__file__).parent

spellcheck_config_filename = ".pyspelling.yml"

# Add spellcheck directory to PYTHONPATH to use custom PySpelling Plugin for Jupyter Notebooks
PYTHONPATH = ":".join([os.environ.get("PYTHONPATH") or "", str(spellcheck_dir)])

# Run PySpelling tool
result = subprocess.run(
    args=["pyspelling", "--config", f"{spellcheck_dir / spellcheck_config_filename}"],
    universal_newlines=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env=dict(os.environ, PYTHONPATH=PYTHONPATH),
)

result_output = result.stdout.strip("\n") if result.stdout else result.stderr.strip("\n")

print(result_output, file=sys.stderr if result.returncode else sys.stdout, flush=True)

exit(result.returncode)

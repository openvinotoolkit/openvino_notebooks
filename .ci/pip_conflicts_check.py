import re
import glob
import subprocess
import sys

for nb_name in glob.glob("./notebooks/*/*.ipynb"):
    print(f"Processing {nb_name}...")
    convert_command = f"jupyter nbconvert --no-prompt --to script --stdout {nb_name}"
    convert_stdout = subprocess.run(convert_command.split(), capture_output=True).stdout
    lines = convert_stdout.decode().split("\n")
    for line in lines:
        matched = re.match(r'^get_ipython\(\).run_line_magic\(\'pip\', \'install (-q )?([^#]*)\'\)', line)
        if matched is not None:
            packages = matched.group(2).replace("\"", "")
            command = f"{sys.executable} -m pip install -q {packages}"
            print(f"    Executing {command}...")
            subprocess.run(command.split(), check=True)
check_command = f"{sys.executable} -m pip check"
print(f"Executing {check_command}...\n\n\n")
subprocess.run(check_command.split(), check=True)

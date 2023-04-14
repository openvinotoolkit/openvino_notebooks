import sys
import os
import subprocess # nosec - disable B404:import-subprocess check
import csv
import shutil
from pathlib import Path
from argparse import ArgumentParser


ROOT = Path(__file__).parents[1]


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--ignore_list', required=False, nargs='+')
    parser.add_argument('--test_list', required=False, nargs='+')
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--report_dir', default='report')
    parser.add_argument('--collect_reports', action='store_true')
    parser.add_argument("--move_notebooks_dir")
    return parser.parse_args()

def find_notebook_dir(path, root):
    for parent in path.parents:
        if root == parent.parent:
            return parent.relative_to(root)
    return None

def move_notebooks(nb_dir):
    current_notebooks_dir = ROOT / 'notebooks'
    shutil.copytree(current_notebooks_dir, nb_dir)         


def prepare_test_plan(test_list, ignore_list, nb_dir=None):
    notebooks_dir = ROOT / 'notebooks' if nb_dir is None else nb_dir
    notebooks = sorted(list(notebooks_dir.rglob('**/*.ipynb')))
    statuses = {notebook.parent.relative_to(notebooks_dir): {'status': '', 'path': notebook.parent} for notebook in notebooks}
    test_list = test_list or statuses.keys()
    if len(test_list) == 1 and test_list[0].endswith('.txt'):
        testing_notebooks = []
        with open(test_list[0], 'r') as f:
            for line in f.readlines():
                changed_path = Path(line.strip())
                if changed_path.resolve() == (ROOT / 'requirements.txt').resolve():
                    print('requirements.txt changed, check all notebooks')
                    testing_notebooks = statuses.keys()
                    break
                if changed_path.suffix == '.md':
                    continue
                notebook_subdir = find_notebook_dir(changed_path.resolve(), notebooks_dir.resolve())
                if notebook_subdir is None:
                    continue
                testing_notebooks.append(notebook_subdir)
        test_list = set(testing_notebooks)
    else:
        test_list = set(map(lambda x: Path(x), test_list))

    ignore_list = ignore_list or []
    ignore_list = set(map(lambda x: Path(x), ignore_list))
    for notebook in statuses:
        if notebook not in test_list:
            statuses[notebook]['status'] = 'SKIPPED'
        if notebook in ignore_list:
            statuses[notebook]['status'] = 'SKIPPED'
    return statuses


def clean_test_artifacts(before_test_files, after_test_files):
    for file_path in after_test_files:
        if file_path in before_test_files or not file_path.exists():
            continue
        if file_path.is_file():
            try:
                file_path.unlink()
            except Exception:
                pass
        else:
            shutil.rmtree(file_path, ignore_errors=True)


def run_test(notebook_path, root):
    print(f'RUN {notebook_path.relative_to(root)}', flush=True)
    
    with cd(notebook_path):
        existing_files = sorted(Path('.').iterdir())
        if not len(existing_files):  # skip empty directories
            return 0
        
        try:
            notebook_name = str([filename for filename in existing_files if str(filename).startswith('test_')][0])
        except IndexError:  # if there is no 'test_' notebook generated
            print('No test_ notebook found.')
            return 0
        
        main_command = [sys.executable,  '-m',  'treon', notebook_name]
        retcode = subprocess.run(main_command).returncode

        clean_test_artifacts(existing_files, sorted(Path('.').iterdir()))
    return retcode


def finalize_status(failed_notebooks, test_plan, report_dir, root):
    return_status = 0
    if failed_notebooks:
        return_status = 1
        print("FAILED: \n{}".format('\n'.join(failed_notebooks)))
    test_report = []
    for notebook, status in test_plan.items():
        test_status = status['status'] or 'NOT_RUN'
        test_report.append({
            'name': notebook, 'status': test_status, 'full_path': str(status['path'].relative_to(root))
        })
    with (report_dir / 'test_report.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'status', 'full_path'])
        writer.writeheader()
        writer.writerows(test_report) 
    return return_status


class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, new_path):
        self.new_path = os.path.expanduser(new_path)

    def __enter__(self):
        self.saved_path = os.getcwd()
        os.chdir(self.new_path)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.saved_path)

def main():
    failed_notebooks = []
    args = parse_arguments()
    reports_dir = Path(args.report_dir)
    reports_dir.mkdir(exist_ok=True, parents=True)
    notebooks_moving_dir = args.move_notebooks_dir
    root = ROOT
    if notebooks_moving_dir is not None:
        notebooks_moving_dir = Path(notebooks_moving_dir)
        root = notebooks_moving_dir.parent
        move_notebooks(notebooks_moving_dir)
    
    test_plan = prepare_test_plan(args.test_list, args.ignore_list, notebooks_moving_dir)
    for notebook, report in test_plan.items():
        if report['status'] == "SKIPPED":
            continue
        status = run_test(report['path'], root)
        report['status'] = 'SUCCESS' if not status else "FAILED"
        if status:
            failed_notebooks.append(str(notebook))
            if args.early_stop:
                break
    exit_status = finalize_status(failed_notebooks, test_plan, reports_dir, root)
    return exit_status


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

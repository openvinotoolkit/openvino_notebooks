import sys
import os
import subprocess
import csv
from pathlib import Path
from argparse import ArgumentParser


ROOT = Path(__file__).parents[1]


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--ignore_list', required=False, nargs='+')
    parser.add_argument('--test_list', required=False, nargs='+')
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--report_dir', default='report')
    return parser.parse_args()


def prepare_test_plan(test_list, ignore_list):
    notebooks = list((ROOT / 'notebooks').rglob('**/test_*.ipynb'))
    statuses = {notebook.parent: {'status': '', 'path': notebook.parent} for notebook in notebooks}
    test_list = test_list or statuses.keys()
    ignore_list = ignore_list or []
    for notebook in statuses:
        if notebook not in test_list:
            statuses[notebook]['status'] = 'SKIPPED'
        if notebook in ignore_list:
            statuses[notebook]['status'] = 'SKIPPED'
    return statuses


def run_test(notebook_path, report_dir):
    print(f'RUN {notebook_path.relative_to(ROOT)}')
    report_file = report_dir / f'{notebook_path.name}_report.xml'
    with cd(notebook_path):
        retcode = subprocess.run([
            sys.executable,  '-m',  'pytest', '--nbval', '-k', 'test_', '--durations', '10', '--junitxml', report_file
            ])
    return retcode


def finalize_status(failed_notebooks, test_plan, report_dir):
    return_status = 0
    if failed_notebooks:
        return_status = 1
        print("FAILED: \n{}".format('\n'.join(failed_notebooks)))
    test_report = []
    for notebook, status in test_plan.items():
        test_status = status['status'] or 'NOT_RUN'
        test_report.append({
            'name': notebook, 'status': test_status, 'full_path': str(status['path'].relative_to(ROOT))
        })
    with (report_dir / 'test_report.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'status', 'full_path'])
        writer.writeheader()
        writer.writerows(test_report) 
    return return_status


import os

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

def main():
    failed_notebooks = []
    args = parse_arguments()
    reports_dir = Path(args.report_dir)
    reports_dir.mkdir(exist_ok=True, parents=True)
    
    test_plan = prepare_test_plan(args.test_list, args.ignore_list)
    for notebook, report in test_plan.items():
        if report['status'] == "SKIPPED":
            continue
        status = run_test(report['path'], reports_dir)
        report['status'] = 'SUCCESS' if not status else "FAILED"
        if not status:
            failed_notebooks.append(notebook)
            if args.early_stop:
                break
    exit_status = finalize_status(failed_notebooks, test_plan, reports_dir)
    return exit_status


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

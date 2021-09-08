from pathlib import Path

def test_readme():
    """
    Test that all notebooks have a README file and exist in the Notebooks README
    """
    notebooks_readme = Path("notebooks/README.md").read_text()
    for item in Path("notebooks").iterdir():
        if item.is_dir():
        # item is a notebook directory
            notebook_dir = item.relative_to("notebooks")
            if str(notebook_dir)[0].isdigit():
                assert "README.md" in [filename.name for filename in item.iterdir()], \
                       f"README not found in {item}"
                assert str(notebook_dir) in notebooks_readme, f"{item} not found in notebooks README"


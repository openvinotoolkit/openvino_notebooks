from pathlib import Path

def test_readme():
    """
    Test that all notebooks have a README file and exist in the Notebooks README
    """
    notebooks_readme = Path("notebooks/README.md").read_text()
    for item in Path("notebooks").iterdir():
        if item.is_dir() and str(item)[0].isdigit():
        # item is a notebook directory
            assert "README.md" in [filename.name for filename in item.iterdir()], \
                   f"README not found in {item}"
            assert str(item) in notebooks_readme, f"{item} not found in notebooks README"

"""
PySpelling plugin for filtering Jupyter Notebook files (*.ipynb)
"""

from pyspelling import filters
import nbformat


class IpynbFilter(filters.Filter):
    """Spellchecking Jupyter Notebook ipynb cells"""

    def __init__(self, options, default_encoding="utf-8"):
        """Initialization."""
        super().__init__(options, default_encoding)

    def get_default_config(self):
        """Get default configuration."""
        return {
            "cell_type": "markdown",  # Cell type to filter (markdown or code)
        }

    def setup(self):
        """Setup."""
        self.cell_type = self.config["cell_type"]

    def filter(self, source_file, encoding):  # noqa A001
        """Open and filter the file from disk."""
        nb: nbformat.NotebookNode = nbformat.read(source_file, as_version=nbformat.NO_CONVERT)

        return [filters.SourceText(self._filter(nb), source_file, encoding, "ipynb")]

    def _filter(self, nb):
        """Filter ipynb."""
        text_list = []
        for cell in nb.cells:
            if cell["cell_type"] == self.cell_type:
                text_list.append(cell["source"])

        return "\n".join(text_list)

    def sfilter(self, source):
        """Execute filter."""
        return [filters.SourceText(self._filter(source.text), source.context, source.encoding, "ipynb")]


def get_plugin():
    """Return the filter."""
    return IpynbFilter

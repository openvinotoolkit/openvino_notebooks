import ast
import json
from pathlib import Path
from types import SimpleNamespace

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "llm-rag-llamaindex" / "llm-rag-llamaindex.ipynb"


def load_create_vectordb():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code" and "def create_vectordb" in source:
            tree = ast.parse(source)
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "create_vectordb")
            namespace = {}
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK_PATH), "exec"), namespace)
            return namespace["create_vectordb"]
    raise AssertionError("create_vectordb not found in the notebook")


def test_create_vectordb_uses_embedding_model_hidden_size():
    """The Gradio document loader must build the FAISS index with the embedding model dimension."""
    dimensions = []

    class FakeIndexFlatL2:
        def __init__(self, dimension):
            dimensions.append(dimension)

    class FakeIndex:
        def as_query_engine(self, **kwargs):
            return SimpleNamespace(query=lambda *args, **kwargs: None)

    create_vectordb = load_create_vectordb()
    create_vectordb.__globals__.update(
        {
            "EMBEDDING_MODEL_HIDDEN_SIZE": 384,
            "faiss": SimpleNamespace(IndexFlatL2=FakeIndexFlatL2),
            "PyMuPDFReader": lambda: SimpleNamespace(load=lambda file_path: ["document"]),
            "TEXT_SPLITERS": {"SentenceSplitter": lambda **kwargs: "splitter"},
            "LangchainNodeParser": lambda splitter: splitter,
            "FaissVectorStore": lambda faiss_index: SimpleNamespace(faiss_index=faiss_index),
            "StorageContext": SimpleNamespace(from_defaults=lambda **kwargs: "storage"),
            "VectorStoreIndex": SimpleNamespace(from_documents=lambda *args, **kwargs: FakeIndex()),
            "reranker": SimpleNamespace(top_n=0),
            "gr": SimpleNamespace(Warning=lambda message: None),
        }
    )

    result = create_vectordb(SimpleNamespace(name="sample.pdf"), "SentenceSplitter", 200, 40, 5, 2, False)

    assert result == "Vector database is Ready"
    assert dimensions == [384]

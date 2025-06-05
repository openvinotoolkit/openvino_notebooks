from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)
from typing import Any, List, Optional, Dict
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager


class OpenVINOGenAIEmbedding(BaseEmbedding):
    model_path: str = Field(description="Local path.")
    max_length: Optional[int] = Field(description="Maximum length of input.")
    pooling: str = Field(description="Pooling strategy. One of ['cls', 'mean'].")
    normalize: bool = Field(default=True, description="Normalize embeddings or not.")
    query_instruction: Optional[str] = Field(description="Instruction to prepend to query text.")
    text_instruction: Optional[str] = Field(description="Instruction to prepend to text.")

    _ov_pipe: Any = PrivateAttr()

    def __init__(
        self,
        model_path: str,
        pooling: str = "cls",
        max_length: Optional[int] = None,
        normalize: bool = True,
        query_instruction: Optional[str] = None,
        text_instruction: Optional[str] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        callback_manager: Optional[CallbackManager] = None,
        model_kwargs: Dict[str, Any] = {},
        device: Optional[str] = "auto",
    ):
        try:
            import openvino_genai

        except ImportError:
            raise ImportError("Could not import OpenVINO GenAI package. " "Please install it with `pip install openvino-genai`.")

        if pooling not in ["cls", "mean"]:
            raise ValueError(f"Pooling {pooling} not supported.")

        config = openvino_genai.TextEmbeddingPipeline.Config()
        config.normalize = normalize
        config.max_length = max_length
        config.pooling_type = (
            openvino_genai.TextEmbeddingPipeline.PoolingType.MEAN if pooling == "mean" else openvino_genai.TextEmbeddingPipeline.PoolingType.CLS
        )
        config.query_instruction = query_instruction
        config.query_instruction = text_instruction

        ov_pipe = openvino_genai.TextEmbeddingPipeline(model_path, device, config, **model_kwargs)

        super().__init__(
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager or CallbackManager([]),
            model_path=model_path,
            max_length=max_length,
            pooling=pooling,
            normalize=normalize,
            query_instruction=query_instruction,
            text_instruction=text_instruction,
        )
        self._device = device
        self._ov_pipe = ov_pipe

    @classmethod
    def class_name(cls) -> str:
        return "OpenVINOGenAIEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._ov_pipe.embed_query(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding async."""
        return self._ov_pipe.embed_query(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding async."""
        return self._ov_pipe.embed_documents([text])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._ov_pipe.embed_documents([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return self._ov_pipe.embed_documents(texts)

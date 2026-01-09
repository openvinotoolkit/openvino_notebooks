from llama_index.core.base.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from typing import Any, List, Optional, Dict
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
from llama_index.core.instrumentation import get_dispatcher


dispatcher = get_dispatcher(__name__)


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


class OpenVINOGenAIReranking(BaseNodePostprocessor):
    model_id_or_path: str = Field(description="Huggingface model id or local path.")
    top_n: int = Field(description="Number of nodes to return sorted by score.")
    keep_retrieval_score: bool = Field(
        default=False,
        description="Whether to keep the retrieval score in metadata.",
    )

    _ov_pipe: Any = PrivateAttr()

    def __init__(
        self,
        model_id_or_path: str,
        max_length: Optional[int] = None,
        top_n: Optional[int] = 3,
        device: Optional[str] = "auto",
        model_kwargs: Dict[str, Any] = {},
        keep_retrieval_score: Optional[bool] = False,
    ):
        try:
            import openvino_genai
        except ImportError:
            raise ImportError("Could not import OpenVINO GenAI package. " "Please install it with `pip install openvino-genai`.")

        super().__init__(top_n=top_n, max_length=max_length, model_id_or_path=model_id_or_path, device=device, keep_retrieval_score=keep_retrieval_score)

        config = openvino_genai.TextRerankPipeline.Config()
        config.top_n = top_n
        if max_length:
            config.max_length = max_length

        ov_pipe = openvino_genai.TextRerankPipeline(model_id_or_path, device, config, **model_kwargs)

        self._ov_pipe = ov_pipe

    @classmethod
    def class_name(cls) -> str:
        return "OpenVINOGenAIReranking"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        dispatcher.event(
            ReRankStartEvent(
                query=query_bundle,
                nodes=nodes,
                top_n=self.top_n,
                model_name=self.model_id_or_path,
            )
        )

        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []

        nodes_text_list = [str(node.node.get_content(metadata_mode=MetadataMode.EMBED)) for node in nodes]

        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.MODEL_NAME: self.model_id_or_path,
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.TOP_K: self.top_n,
            },
        ) as event:
            outputs = self._ov_pipe.rerank(query_bundle.query_str, nodes_text_list)

            for node, score in zip(nodes, outputs):
                if self.keep_retrieval_score:
                    # keep the retrieval score in metadata
                    node.node.metadata["retrieval_score"] = node.score
                node.score = score

            event.on_end(payload={EventPayload.NODES: nodes})

        dispatcher.event(ReRankEndEvent(nodes=nodes))
        return nodes

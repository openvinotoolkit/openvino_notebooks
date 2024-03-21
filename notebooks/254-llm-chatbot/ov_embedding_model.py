from langchain.pydantic_v1 import BaseModel, Extra, Field
from langchain.schema.embeddings import Embeddings
from typing import Optional, Union, Dict,  Any, List
from transformers import AutoTokenizer
import torch
from tqdm.autonotebook import trange
import numpy as np
from optimum.intel.openvino import OVModelForFeatureExtraction
from numpy import ndarray


import logging
logger = logging.getLogger(__name__)

DEFAULT_QUERY_BGE_INSTRUCTION_EN = (
    "Represent this question for searching relevant passages: "
)
DEFAULT_QUERY_BGE_INSTRUCTION_ZH = "为这个句子生成表示以用于检索相关文章："


class OVBgeEmbeddings(BaseModel, Embeddings):
    """HuggingFace BGE sentence_transformers embedding models.

    To use, you should have the ``sentence_transformers`` python package installed.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import HuggingFaceBgeEmbeddings

            model_name = "BAAI/bge-large-en"
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': True}
            hf = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
    """

    ov_model: Any
    tokenizer: Any
    model_dir: str
    """Model name to use."""
    cache_folder: Optional[str] = None
    """Path to store models.
    Can be also set by SENTENCE_TRANSFORMERS_HOME environment variable."""
    device: str = "CPU"
    ov_config: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass to the model."""
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method of the model."""
    query_instruction: str = DEFAULT_QUERY_BGE_INSTRUCTION_EN
    """Instruction to use for embedding query."""

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)

        self.ov_model = OVModelForFeatureExtraction.from_pretrained(
            self.model_dir, device=self.device, ov_config=self.ov_config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        if "-zh" in self.model_dir:
            self.query_instruction = DEFAULT_QUERY_BGE_INSTRUCTION_ZH

    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):  # Object has no len() method
            return 1
        # Empty string or list of ints
        elif len(text) == 0 or isinstance(text[0], int):
            return len(text)
        else:
            # Sum of length of individual strings
            return sum([len(t) for t in text])

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 4,
        show_progress_bar: bool = None,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        normalize_embeddings: bool = False,
    ) -> Union[List[Tensor], ndarray, Tensor]:
        """
        Computes sentence embeddings.

        :param sentences: the sentences to embed.
        :param prompt_name: The name of the prompt to use for encoding. Must be a key in the `prompts` dictionary,
            which is either set in the constructor or loaded from the model configuration. For example if
            `prompt_name` is ``"query"`` and the `prompts` is ``{"query": "query: ", ...}``, then the sentence "What
            is the capital of France?" will be encoded as "query: What is the capital of France?" because the sentence
            is appended to the prompt. If `prompt` is also set, this argument is ignored.
        :param prompt: The prompt to use for encoding. For example, if the prompt is ``"query: "``, then the
            sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?"
            because the sentence is appended to the prompt. If `prompt` is set, `prompt_name` is ignored.
        :param batch_size: the batch size used for the computation.
        :param show_progress_bar: Whether to output a progress bar when encode sentences.
        :param output_value: The type of embeddings to return: "sentence_embedding" to get sentence embeddings,
            "token_embeddings" to get wordpiece token embeddings, and `None`, to get all output values. Defaults
            to "sentence_embedding".
        :param convert_to_numpy: Whether the output should be a list of numpy vectors. If False, it is a list of PyTorch tensors.
        :param convert_to_tensor: Whether the output should be one large tensor. Overwrites `convert_to_numpy`.
        :param device: Which `torch.device` to use for the computation.
        :param normalize_embeddings: Whether to normalize returned vectors to have length 1. In that case,
            the faster dot-product (util.dot_score) instead of cosine similarity can be used.

        :return: By default, a 2d numpy array with shape [num_inputs, output_dimension] is returned. If only one string
            input is provided, then the output is a 1d array with shape [output_dimension]. If `convert_to_tensor`, a
            torch Tensor is returned instead.
        """

        if convert_to_tensor:
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(
            sentences, "__len__"
        ):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        all_embeddings = []
        length_sorted_idx = np.argsort(
            [-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index: start_index + batch_size]
            features = self.tokenizer(
                sentences_batch, padding=True, truncation=True,  return_tensors='pt')

            out_features = self.ov_model(**features)
            embeddings = out_features[0][:, 0]
            if normalize_embeddings:
                embeddings = torch.nn.functional.normalize(
                    embeddings, p=2, dim=1)

            # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
            if convert_to_numpy:
                embeddings = embeddings.cpu()

            all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx]
                          for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            if len(all_embeddings):
                all_embeddings = torch.stack(all_embeddings)
            else:
                all_embeddings = torch.Tensor()
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy()
                                        for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = [t.replace("\n", " ") for t in texts]
        embeddings = self.encode(texts, **self.encode_kwargs)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        embedding = self.encode(
            self.query_instruction + text, **self.encode_kwargs
        )
        return embedding.tolist()

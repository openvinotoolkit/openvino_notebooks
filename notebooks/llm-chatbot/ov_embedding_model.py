from langchain.pydantic_v1 import BaseModel, Extra, Field
from langchain.schema.embeddings import Embeddings
from typing import Optional, Union, Dict, Tuple, Any, List
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer
from pathlib import Path
import openvino as ov
import torch
import numpy as np

class OVEmbeddings(BaseModel, Embeddings):
    """
    LangChain compatible model wrapper for embedding model
    """

    model: Any  #: :meta private:
    """LLM Transformers model."""
    model_kwargs: Optional[dict] = None
    """OpenVINO model configurations."""
    tokenizer: Any  #: :meta private:
    """Huggingface tokenizer model."""
    do_norm: bool
    """Whether normlizing the output of model"""
    num_stream: int
    """Number of stream."""
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method of the model."""

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        do_norm: bool,
        ov_config: Optional[dict],
        model_kwargs: Optional[dict],
        **kwargs: Any,
    ):
        _model_kwargs = model_kwargs or {}
        _ov_config = ov_config or {}
        tokenizer = AutoTokenizer.from_pretrained(model_id, **_model_kwargs)
        core = ov.Core()
        model_path = Path(model_id) / "openvino_model.xml"
        model = core.compile_model(model_path, **_ov_config)
        num_stream = model.get_property('NUM_STREAMS')

        return cls(
            model=model,
            tokenizer=tokenizer,
            do_norm = do_norm,
            num_stream=num_stream,
            **kwargs,
        )

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

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

    def encode(self, sentences: Union[str, List[str]]):
        """
        Computes sentence embeddings

        Args: 
            sentences: the sentences to embed

        Returns:
           By default, a list of tensors is returned.
        """
        all_embeddings = []
        length_sorted_idx = np.argsort(
            [-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        nireq = self.num_stream + 1
        infer_queue = ov.AsyncInferQueue(self.model, nireq)

        def postprocess(request, userdata):
            embeddings = request.get_output_tensor(0).data
            embeddings = np.mean(embeddings, axis=1)
            if self.do_norm:
                embeddings = normalize(embeddings, 'l2')
            all_embeddings.extend(embeddings)

        infer_queue.set_callback(postprocess)
        
        for i, sentence in enumerate(sentences_sorted):
            inputs = {}
            features = self.tokenizer(
                sentence, padding=True, truncation=True, return_tensors='np')
            for key in features:
                inputs[key] = features[key]
            infer_queue.start_async(inputs, i)
        infer_queue.wait_all()
        all_embeddings = np.asarray(all_embeddings)
        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = self.encode(texts, **self.encode_kwargs)

        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]

import numpy as np
from pathlib import Path
from optimum.intel.openvino import OVModelForSequenceClassification
from transformers import AutoTokenizer
from langchain.callbacks.manager import Callbacks
from langchain_core.documents import Document
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from typing import Optional, Union, Dict, Tuple, Any, List, TYPE_CHECKING, Sequence
from langchain.pydantic_v1 import BaseModel, Extra, Field
import json
from pathlib import Path
from tokenizers import AddedToken, Tokenizer
import onnxruntime as ort
import numpy as np
import os
import zipfile
import requests
from tqdm import tqdm
import collections

class RerankRequest:

    def __init__(self, query=None, passages=None):
        self.query = query
        self.passages = passages if passages is not None else []

class OVRanker(BaseDocumentCompressor):
    
    ov_model: Any
    tokenizer: Any 
    model_dir: str
    device: str = "CPU"
    ov_config: Dict[str, Any] = Field(default_factory=dict)
    top_n: int = 4
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.tokenizer = self._get_tokenizer()
        self.ov_model = OVModelForSequenceClassification.from_pretrained(self.model_dir, device=self.device, ov_config=self.ov_config)
        
    def _load_vocab(self, vocab_file):
    
        vocab = collections.OrderedDict()
        with open(vocab_file, "r", encoding="utf-8") as reader:
            tokens = reader.readlines()
        for index, token in enumerate(tokens):
            token = token.rstrip("\n")
            vocab[token] = index
        return vocab

    def _get_tokenizer(self, max_length = 512):
      
        config_path = Path(self.model_dir) / "config.json"
        if not config_path.exists():
          raise FileNotFoundError(f"config.json missing in {self.model_dir}")
        
        tokenizer_path = Path(self.model_dir) / "tokenizer.json"
        if not tokenizer_path.exists():
          raise FileNotFoundError(f"tokenizer.json missingin  {self.model_dir}")
        
        tokenizer_config_path = Path(self.model_dir) / "tokenizer_config.json"
        if not tokenizer_config_path.exists():
          raise FileNotFoundError(f"tokenizer_config.json missing in  {Path(self.model_dir)}")
        
        tokens_map_path = Path(self.model_dir) / "special_tokens_map.json"
        if not tokens_map_path.exists():
          raise FileNotFoundError(f"special_tokens_map.json missing in  {Path(self.model_dir)}")
        
        config = json.load(open(str(config_path)))
        tokenizer_config = json.load(open(str(tokenizer_config_path)))
        tokens_map = json.load(open(str(tokens_map_path)))
        
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer.enable_truncation(max_length=min(tokenizer_config["model_max_length"], max_length))
        tokenizer.enable_padding(pad_id=config["pad_token_id"], pad_token=tokenizer_config["pad_token"])
        
        for token in tokens_map.values():
          if isinstance(token, str):
              tokenizer.add_special_tokens([token])
          elif isinstance(token, dict):
              tokenizer.add_special_tokens([AddedToken(**token)])
        
        vocab_file = Path(self.model_dir) / "vocab.txt"
        if vocab_file.exists():
          tokenizer.vocab = self._load_vocab(vocab_file)
          tokenizer.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in tokenizer.vocab.items()])                
        
        return tokenizer
    
    def rerank(self, request):
        query = request.query
        passages = request.passages

        query_passage_pairs = [[query, passage["text"]] for passage in passages]
        input_text = self.tokenizer.encode_batch(query_passage_pairs)
        input_ids = [e.ids for e in input_text]
        token_type_ids = [e.type_ids for e in input_text]
        attention_mask = [e.attention_mask for e in input_text]
        
        use_token_type_ids = token_type_ids is not None and not np.all(token_type_ids == 0)

        if use_token_type_ids:
            input_tensors = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        else:
            input_tensors = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }


        # input_data = {k: v for k, v in onnx_input.items()}
        print(input_tensors)
        outputs = self.ov_model(**input_tensors, return_dict=True)
        print(outputs)
        if outputs[0].shape[1] > 1:
            scores = outputs[0][:, 1]
        else:
            scores = outputs[0].flatten()

        scores = list(1 / (1 + np.exp(-scores)))  

        # Combine scores with passages, including metadata
        for score, passage in zip(scores, passages):
            passage["score"] = score

        # Sort passages based on scores
        passages.sort(key=lambda x: x["score"], reverse=True)

        return passages
    
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        passages = [
            {"id": i, "text": doc.page_content} for i, doc in enumerate(documents)
        ]

        rerank_request = RerankRequest(query=query, passages=passages)
        rerank_response = self.rerank(rerank_request)[: self.top_n]
        final_results = []
        for r in rerank_response:
            doc = Document(
                page_content=r["text"],
                metadata={"id": r["id"], "relevance_score": r["score"]},
            )
            final_results.append(doc)
        return final_results
    
from pathlib import Path
from typing import Optional
import os
import torch
import huggingface_hub as hf_hub
from transformers import AutoModel, AutoConfig
import collections
import unicodedata
from transformers.tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
import openvino as ov
import nncf
import cv2
import numpy as np
import shutil
import types
import gc

VISION_ENCODER_NAME = "openvino_vision_encoder.xml"
TEXT_ENCODER_NAME = "openvino_text_encoder.xml"


def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


class DictToClass:
    def __init__(self, data):
        for key, value in data.items():
            key = str(key)
            if isinstance(value, dict):
                setattr(self, key, DictToClass(value))
            elif isinstance(value, list):
                setattr(self, key, [DictToClass(item) if isinstance(item, dict) else item for item in value])
            else:
                setattr(self, key, value)

    def __repr__(self):
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"


def _frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break


v_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)


def normalize(data):
    return (data / 255.0 - v_mean) / v_std


def frames2tensor(vid_list, fnum=8, target_size=(224, 224), device=torch.device("cuda")):
    assert len(vid_list) >= fnum
    step = len(vid_list) // fnum
    vid_list = vid_list[::step][:fnum]
    vid_list = [cv2.resize(x[:, :, ::-1], target_size) for x in vid_list]
    vid_tube = [np.expand_dims(normalize(x), axis=(0, 1)) for x in vid_list]
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
    return vid_tube


def vid2tensor(path: str, fnum: int = 8, target_size: tuple = (224, 224), device=torch.device("cuda")):
    video = cv2.VideoCapture(path)
    frames = [x for x in _frame_from_video(video)]
    return frames2tensor(frames, fnum, target_size, device)


def get_text_feat_dict(texts, clip, text_feat_d={}):
    for t in texts:
        feat = clip.get_txt_feat(t)
        text_feat_d[t] = feat
    return text_feat_d


def get_vid_feat(frames, vlm):
    return vlm.get_vid_features(frames)


def retrieve_text(frames, texts, vlm, topk: int = 5):

    config = vlm._config

    fn = config.num_frames
    size_t = config.size_t
    frames_tensor = frames2tensor(frames, fnum=fn, target_size=(size_t, size_t), device=torch.device("cpu"))
    vid_feat = vlm.get_vid_feat(frames_tensor)

    text_feat_d = {}
    text_feat_d = get_text_feat_dict(texts, vlm, text_feat_d)
    text_feats = [text_feat_d[t] for t in texts]
    text_feats_tensor = torch.cat(text_feats, 0)

    probs, idxs = vlm.predict_label(vid_feat, text_feats_tensor, top=topk)

    ret_texts = [texts[i] for i in idxs.long().numpy()[0].tolist()]
    return ret_texts, probs.float().numpy()[0]


VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "bert-base-uncased": "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt",
        "bert-large-uncased": "https://huggingface.co/bert-large-uncased/resolve/main/vocab.txt",
        "bert-base-cased": "https://huggingface.co/bert-base-cased/resolve/main/vocab.txt",
        "bert-large-cased": "https://huggingface.co/bert-large-cased/resolve/main/vocab.txt",
        "bert-base-multilingual-uncased": "https://huggingface.co/bert-base-multilingual-uncased/resolve/main/vocab.txt",
        "bert-base-multilingual-cased": "https://huggingface.co/bert-base-multilingual-cased/resolve/main/vocab.txt",
        "bert-base-chinese": "https://huggingface.co/bert-base-chinese/resolve/main/vocab.txt",
        "bert-base-german-cased": "https://huggingface.co/bert-base-german-cased/resolve/main/vocab.txt",
        "bert-large-uncased-whole-word-masking": "https://huggingface.co/bert-large-uncased-whole-word-masking/resolve/main/vocab.txt",
        "bert-large-cased-whole-word-masking": "https://huggingface.co/bert-large-cased-whole-word-masking/resolve/main/vocab.txt",
        "bert-large-uncased-whole-word-masking-finetuned-squad": "https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/vocab.txt",
        "bert-large-cased-whole-word-masking-finetuned-squad": "https://huggingface.co/bert-large-cased-whole-word-masking-finetuned-squad/resolve/main/vocab.txt",
        "bert-base-cased-finetuned-mrpc": "https://huggingface.co/bert-base-cased-finetuned-mrpc/resolve/main/vocab.txt",
        "bert-base-german-dbmdz-cased": "https://huggingface.co/bert-base-german-dbmdz-cased/resolve/main/vocab.txt",
        "bert-base-german-dbmdz-uncased": "https://huggingface.co/bert-base-german-dbmdz-uncased/resolve/main/vocab.txt",
        "TurkuNLP/bert-base-finnish-cased-v1": "https://huggingface.co/TurkuNLP/bert-base-finnish-cased-v1/resolve/main/vocab.txt",
        "TurkuNLP/bert-base-finnish-uncased-v1": "https://huggingface.co/TurkuNLP/bert-base-finnish-uncased-v1/resolve/main/vocab.txt",
        "wietsedv/bert-base-dutch-cased": "https://huggingface.co/wietsedv/bert-base-dutch-cased/resolve/main/vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "bert-base-uncased": 512,
    "bert-large-uncased": 512,
    "bert-base-cased": 512,
    "bert-large-cased": 512,
    "bert-base-multilingual-uncased": 512,
    "bert-base-multilingual-cased": 512,
    "bert-base-chinese": 512,
    "bert-base-german-cased": 512,
    "bert-large-uncased-whole-word-masking": 512,
    "bert-large-cased-whole-word-masking": 512,
    "bert-large-uncased-whole-word-masking-finetuned-squad": 512,
    "bert-large-cased-whole-word-masking-finetuned-squad": 512,
    "bert-base-cased-finetuned-mrpc": 512,
    "bert-base-german-dbmdz-cased": 512,
    "bert-base-german-dbmdz-uncased": 512,
    "TurkuNLP/bert-base-finnish-cased-v1": 512,
    "TurkuNLP/bert-base-finnish-uncased-v1": 512,
    "wietsedv/bert-base-dutch-cased": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "bert-base-uncased": {"do_lower_case": True},
    "bert-large-uncased": {"do_lower_case": True},
    "bert-base-cased": {"do_lower_case": False},
    "bert-large-cased": {"do_lower_case": False},
    "bert-base-multilingual-uncased": {"do_lower_case": True},
    "bert-base-multilingual-cased": {"do_lower_case": False},
    "bert-base-chinese": {"do_lower_case": False},
    "bert-base-german-cased": {"do_lower_case": False},
    "bert-large-uncased-whole-word-masking": {"do_lower_case": True},
    "bert-large-cased-whole-word-masking": {"do_lower_case": False},
    "bert-large-uncased-whole-word-masking-finetuned-squad": {"do_lower_case": True},
    "bert-large-cased-whole-word-masking-finetuned-squad": {"do_lower_case": False},
    "bert-base-cased-finetuned-mrpc": {"do_lower_case": False},
    "bert-base-german-dbmdz-cased": {"do_lower_case": False},
    "bert-base-german-dbmdz-uncased": {"do_lower_case": True},
    "TurkuNLP/bert-base-finnish-cased-v1": {"do_lower_case": False},
    "TurkuNLP/bert-base-finnish-uncased-v1": {"do_lower_case": True},
    "wietsedv/bert-base-dutch-cased": {"do_lower_case": False},
}


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class BasicTokenizer(object):
    """
    Constructs a BasicTokenizer that will run basic tokenization (punctuation splitting, lower casing, etc.).
    Args:
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to lowercase the input when tokenizing.
        never_split (:obj:`Iterable`, `optional`):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            :obj:`do_basic_tokenize=True`
        tokenize_chinese_chars (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to tokenize Chinese characters.
            This should likely be deactivated for Japanese (see this `issue
            <https://github.com/huggingface/transformers/issues/328>`__).
        strip_accents: (:obj:`bool`, `optional`):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for :obj:`lowercase` (as in the original BERT).
    """

    def __init__(self, do_lower_case=True, never_split=None, tokenize_chinese_chars=True, strip_accents=None):
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents

    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. Split on "white spaces" only, for sub-word tokenization, see
        WordPieceTokenizer.
        Args:
            **never_split**: (`optional`) list of str
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                :func:`PreTrainedTokenizer.tokenize`) List of token not to split.
        """
        # union() returns a new set by concatenating the two sets.
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if token not in never_split:
                if self.do_lower_case:
                    token = token.lower()
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        if never_split is not None and text in never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform
        tokenization using the given vocabulary.
        For example, :obj:`input = "unaffable"` wil return as output :obj:`["un", "##aff", "##able"]`.
        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.
        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


class BertTokenizer(PreTrainedTokenizer):
    r"""
    Construct a BERT tokenizer. Based on WordPiece.
    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.
    Args:
        vocab_file (:obj:`str`):
            File containing the vocabulary.
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to lowercase the input when tokenizing.
        do_basic_tokenize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to do basic tokenization before WordPiece.
        never_split (:obj:`Iterable`, `optional`):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            :obj:`do_basic_tokenize=True`
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (:obj:`str`, `optional`, defaults to :obj:`"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (:obj:`str`, `optional`, defaults to :obj:`"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (:obj:`str`, `optional`, defaults to :obj:`"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        tokenize_chinese_chars (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to tokenize Chinese characters.
            This should likely be deactivated for Japanese (see this `issue
            <https://github.com/huggingface/transformers/issues/328>`__).
        strip_accents: (:obj:`bool`, `optional`):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for :obj:`lowercase` (as in the original BERT).
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs,
    ):
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file)
            )
        self.vocab = load_vocab(vocab_file)

        super().__init__(
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)

    @property
    def do_lower_case(self):
        return self.basic_tokenizer.do_lower_case

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text):
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):

                # If the token is part of the never_split set
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(self, token_ids_0: list[int], token_ids_1: Optional[list[int]] = None) -> list[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:
        - single sequence: ``[CLS] X ``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(self, token_ids_0: list[int], token_ids_1: Optional[list[int]] = None, already_has_special_tokens: bool = False) -> list[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` method.
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the token list is already formatted with special tokens for the model.
        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of " "ids is already formatted with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0: list[int], token_ids_1: Optional[list[int]] = None) -> list[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
        pair mask has the following format:
        ::
            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |
        If :obj:`token_ids_1` is :obj:`None`, this method only returns the first portion of the mask (0s).
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of `token type IDs <../glossary.html#token-type-ids>`_ according to the given
            sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple[str]:
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"])
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    print(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(vocab_file)
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)


def patch_model_code(model_dir):
    modeling_file = model_dir / "modeling_internvideo2.py"
    orig_modeling_file = model_dir / "orig_modeling_internvideo2.py"
    if not orig_modeling_file.exists():
        modeling_file.rename(orig_modeling_file)
        with orig_modeling_file.open("r", encoding="utf-8") as in_f:
            content = in_f.read()
            content = content.replace(
                "self.tokenizer = BertTokenizer.from_pretrained(self._config.model.text_encoder.pretrained, local_files_only=True, use_safetensors=True)",
                "self.tokenizer = BertTokenizer.from_pretrained(self._config.model.text_encoder.pretrained, use_safetensors=True)",
            )
            content = content.replace(
                "from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func\nfrom flash_attn.bert_padding import unpad_input, pad_input",
                "try:\n    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func\n    from flash_attn.bert_padding import unpad_input, pad_input\n    flash_attn_available=True\nexcept:\n    flash_attn_available = False",
            )
            content = content.replace("self.use_flash_attn = use_flash_attn", "self.use_flash_attn = use_flash_attn and flash_attn_available")
            with modeling_file.open("w", encoding="utf-8") as out_f:
                out_f.write(content)
    orig_config_file = model_dir / "orig_config.json"
    config_file = model_dir / "config.json"
    if not orig_config_file.exists():
        config_file.rename(orig_config_file)
        with orig_config_file.open("r", encoding="utf-8") as in_f:
            content = in_f.read()
            configs_dir = model_dir / "configs"
            content = content.replace('"configs/', f'"{configs_dir.absolute()}/')
            with config_file.open("w", encoding="utf-8") as out_f:
                out_f.write(content)


def convert_internvideo(model_id, output_dir, weights_compression_config=None):
    output_dir = Path(output_dir)
    model_dir = Path(model_id.split("/")[-1])
    if not (output_dir / VISION_ENCODER_NAME).exists() or not (output_dir / TEXT_ENCODER_NAME).exists():
        output_dir.mkdir(exist_ok=True, parents=True)
        print(f"⌛ {model_id} conversion started. Be patient, it may takes some time.")
        print("⌛ Load Original model")
        if not model_dir.exists():
            hf_hub.snapshot_download(model_id, local_dir=model_dir)
            patch_model_code(model_dir)

        shutil.copy(model_dir / "config.json", output_dir / "config.json")
        shutil.copytree(model_dir / "configs", output_dir / "configs", dirs_exist_ok=True)
        shutil.copy(model_dir / "modeling_internvideo2.py", output_dir / "modeling_internvideo2.py")
        model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).eval()
        model._config.device = torch.device("cpu")
        model.tokenizer.save_pretrained(output_dir)
        print("✅ Original model successfully loaded")

        if not (output_dir / VISION_ENCODER_NAME).exists():
            print("⌛ Convert Vision Encoder model")

            model.forward = model.get_vid_feat

            def _naive_attn(self, x):
                B, N, C = x.shape
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

                if self.qk_normalization:
                    B_, H_, N_, D_ = q.shape
                    q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
                    k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
                attention_mask = torch.zeros(q.size(-2), k.size(-2))
                x = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=self.scale, is_causal=False, attn_mask=attention_mask)
                x = x.transpose(1, 2).reshape(B, N, C)
                x = self.proj(x)
                x = self.proj_drop(x)
                return x

            def cross_attn_forward(self, x, k=None, v=None):
                B, N, C = x.shape
                N_k = k.shape[1]
                N_v = v.shape[1]

                q_bias, k_bias, v_bias = None, None, None
                if self.q_bias is not None:
                    q_bias = self.q_bias
                    k_bias = self.k_bias
                    v_bias = self.v_bias

                q = torch.nn.functional.linear(input=x, weight=self.q.weight, bias=q_bias)
                q = q.reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)  # (B, N_head, N_q, dim)

                k = torch.nn.functional.linear(input=k, weight=self.k.weight, bias=k_bias)
                k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

                v = torch.nn.functional.linear(input=v, weight=self.v.weight, bias=v_bias)
                v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

                x = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=self.scale, is_causal=False)

                x = x.transpose(1, 2).reshape(B, N, -1)
                x = self.proj(x)
                x = self.proj_drop(x)

                return x

            for block in model.vision_encoder.blocks:
                block.with_cp = False
                block.attn._naive_attn = types.MethodType(_naive_attn, block.attn)

            model.vision_encoder.clip_projector.cross_attn.forward = types.MethodType(cross_attn_forward, model.vision_encoder.clip_projector.cross_attn)
            vision_encoder = ov.convert_model(model, example_input=torch.zeros([1, 4, 3, 224, 224]))
            if weights_compression_config is not None:
                vision_encoder = nncf.compress_weights(vision_encoder, **weights_compression_config)
            ov.save_model(vision_encoder, output_dir / VISION_ENCODER_NAME)
            del vision_encoder
            cleanup_torchscript_cache()
            del model.vision_encoder
            gc.collect()
            print("✅ Vision Encoder model successfully converted")

        if not (output_dir / TEXT_ENCODER_NAME).exists():
            print("⌛ Convert Text Encoder model")

            def forward(self, input_ids, attention_mask):
                text_output = self.get_text_encoder()(
                    input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    mode="text",
                )
                text_embeds = text_output.last_hidden_state
                pooled_text_embeds = text_embeds[:, 0]
                tfeat = self.text_proj(pooled_text_embeds)
                return tfeat / tfeat.norm(dim=-1, keepdim=True)

            model.forward = types.MethodType(forward, model)

            attention_mask = torch.ones([2, 40], dtype=torch.long)
            attention_mask[:, -10:] = 0
            text_encoder = ov.convert_model(model, example_input={"input_ids": torch.ones([2, 40], dtype=torch.long), "attention_mask": attention_mask})

            ov.save_model(text_encoder, output_dir / TEXT_ENCODER_NAME)
            print("✅ Text Encoder model successfully converted")
        print(f"✅ {model_id} model conversion finished. You can find results in {output_dir}")
    else:
        print(f"✅ {model_id} model already converted. You can find results in {output_dir}")


core = ov.Core()


class OVInternVideoStage2:
    def __init__(self, model_dir, device_map="CPU", ov_config=None):
        model_dir = Path(model_dir)
        if isinstance(device_map, str):
            device = device_map
            device_map = {"text_encoder": device.upper(), "vision_encoder": device.upper(), "text_proj": device.upper()}
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True).to_dict()
        self._config = DictToClass(config) if isinstance(config, dict) else config
        self._config.device = torch.device("cpu")
        self.tokenizer = BertTokenizer.from_pretrained(self._config.model.text_encoder.pretrained, use_safetensors=True)
        self.text_encoder = core.compile_model(model_dir / TEXT_ENCODER_NAME, device_map["text_encoder"], config=ov_config)
        self.vision_encoder = core.compile_model(model_dir / VISION_ENCODER_NAME, device_map["vision_encoder"], config=ov_config)

    def get_vid_feat(self, frames: torch.Tensor):
        """get the video features for the given frames.

        Args:
            frames (torch.Tensor): The input frames. Shape: [B,T,C,H,W].

        Returns: tuple.
            - pooled_vision_embeds (torch.Tensor): The pooled output features. Shape: [B,1,C].

        """
        vfeat = self.vision_encoder(frames)[0]
        return torch.from_numpy(vfeat)

    def get_txt_feat(self, text: str):
        """get the text features for the given text."""
        text = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self._config.max_txt_l,
            return_tensors="pt",
        ).to(self._config.device)
        return torch.from_numpy(self.text_encoder({"input_ids": text.input_ids, "attention_mask": text.attention_mask})[0])

    def predict_label(self, vid_feat: torch.Tensor, txt_feat: torch.Tensor, top: int = 5):
        label_probs = (100 * vid_feat @ txt_feat.T).softmax(dim=-1)
        top_probs, top_labels = label_probs.float().cpu().topk(top, dim=-1)
        return top_probs, top_labels

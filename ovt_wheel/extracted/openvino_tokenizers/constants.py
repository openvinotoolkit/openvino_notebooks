# -*- coding: utf-8 -*-
# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from enum import Enum


ATTENTION_MASK_INPUT_NAME = "attention_mask"
TOKEN_IDS_INPUT_NAME = "input_ids"
TOKEN_TYPE_IDS_INPUT_NAME = "token_type_ids"

LOGITS_OUTPUT_NAME = "logits"
TOKEN_IDS_OUTPUT_NAME = "token_ids"
STRING_OUTPUT_NAME = "string_output"

BOS_TOKEN_ID_NAME = "bos_token_id"
EOS_TOKEN_ID_NAME = "eos_token_id"
PAD_TOKEN_ID_NAME = "pad_token_id"
CHAT_TEMPLATE_NAME = "chat_template"
ORIGINAL_TOKENIZER_CLASS_NAME = "original_tokenizer_class"
ORIGINAL_POST_PROCESSOR_NAME = "original_post_processor_template"
PROCESSED_POST_PROCESSOR_NAME = "processed_post_processor_template"

rt_info_to_hf_attribute_map = {
    BOS_TOKEN_ID_NAME: (BOS_TOKEN_ID_NAME,),
    EOS_TOKEN_ID_NAME: (EOS_TOKEN_ID_NAME, "eod_id"),
    PAD_TOKEN_ID_NAME: (PAD_TOKEN_ID_NAME,),
    CHAT_TEMPLATE_NAME: (CHAT_TEMPLATE_NAME, "default_chat_template"),
}

GREEDY_DECODER_NAME = "greedy_decoder"

TOKENIZER_NAME = "tokenizer"
DETOKENIZER_NAME = "detokenizer"

MIN_CACHE_CAPACITY = 20_000
VOCAB_SIZE_CACHE_PROPORTION = 0.2


class UTF8ReplaceMode(Enum):
    IGNORE: str = "ignore"
    REPLACE: str = "replace"
    DISABLE: str = "disable"

    def __str__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, (UTF8ReplaceMode)):
            # UTF8ReplaceMode is a singleton, so we can compare them by reference
            return self is other
        elif isinstance(other, str):
            return self.value == other
        else:
            return False

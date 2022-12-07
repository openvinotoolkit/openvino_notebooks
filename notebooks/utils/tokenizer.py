#!/usr/bin/env python
# coding: utf-8

import numpy as np
import unicodedata
import collections

# Loads a vocabulary file into a dictionary.


def load_vocab(vocab_file):
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip('\n')
        vocab[token] = index
    return vocab


# Location of the local vocab file
file = f'{"data/bert-large-uncased-whole-word-masking-vocab.txt"}'
vocab = load_vocab(file)

# Runs basic whitespace cleaning and splitting on a piece of text.


def whitespace_tokenize(text):
    text = text.lower()
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

# Returns the word from a given id


def tokens_to_ids(token):
    ids = list(vocab.keys())[token]
    return ids

# Returns the id of a given word


def word_to_token(word):
    return vocab[word]

# Define the special tokens of the vocab


def special_tokens_list():
    special_tokens = {
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "mask_token": "[MASK]"}
    return special_tokens

# Split the sentence into on punctuation, except if it is a special token


def _run_split_on_punc(text):
    special_tokens = special_tokens_list()
    for i in list(special_tokens.values()):
        if text == i.casefold():
            return i
    else:
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            punc = unicodedata.category(char).startswith("P")
            if punc:
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

    return ["".join(x) for x in output]


"""
Check if character is a space
(including tab, carriage return and newline characters)
"""


def _is_whitespace(char):
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False

# Returns how many mask tokens exist in a sentence


def check_mask_token(text):
    mask_exists = -1
    special_tokens = special_tokens_list()
    words = text.split()
    for i in range(0, len(words)):
        if words[i] == special_tokens["mask_token"]:
            mask_exists = mask_exists + 1

    return mask_exists


"""
Tokenize a sentence
1. If a particular word isn't found in the vocab,
    it is recursively divided into subwords and
    checked in the vocab till a match is found.
2. Finally padding is applied, if mentioned.
    The default padding is 0, i.e. no padding is applied.
    This final array is called input_ids
3. Token type ids is created for the input sentence
    which is an array of 0s.
4. Attention mask is created which is 1
    for all the non zero tokens of input ids.
5. Finally, input ids, Attention mask and Token type ids
    are combined in a dictionary to be returned
    as an input to the notebook.
"""


def tokenize(text1, padding=0):
    # print(text1)
    token = []
    for word in text1:
        if word in vocab.keys():
            token.append(word_to_token(word))
        else:
            s = 0
            e = len(word)
            n = len(word)
            while (s < e):
                e = e-1
                subword = word[s:e]
                if subword in vocab.keys():
                    token.append(word_to_token(subword))
                    nextsubword = "##" + word[e:n]
                    # print(nextsubword)
                    token.append(word_to_token(nextsubword))
                    break
    if padding != 0:
        input_ids = np.pad(token, [0, padding-len(token)], constant_values=0)

    input_ids = np.array([input_ids.tolist()]).astype(np.int32)
    token_type_ids = np.array(
        [np.zeros(padding, dtype="int32").tolist()]).astype(np.int32)
    a1 = np.ones(len(token), dtype="int32").tolist()
    a2 = np.zeros(padding-len(token), dtype="int32").tolist()
    a1.extend(a2)
    attention_mask = np.array([a1]).astype(np.int32)

    inputs = dict({"input_ids": input_ids, "token_type_ids": token_type_ids,
                  "attention_mask": attention_mask})
    return inputs


"""
Preprocess the text:
1. Add the [CLS] token in the beginning to denote the start of a new sentence
2. If no [MASK] token exists add the mask token at the end.
3. Add [SEP] token to mark the end of the sentence
4. Finally convert the words to their respective tokens
   and return the token array
"""


def preprocess_text(text, padding=0):
    special_tokens = special_tokens_list()
    mask_exists = check_mask_token(text)
    if mask_exists > 0:
        return -1
    else:
        text = whitespace_tokenize(text)

        text1 = []
        for i in text:
            tmp = _run_split_on_punc(i)

            if (isinstance(tmp, collections.abc.Sequence) and
               tmp not in list(special_tokens.values())):
                text1.extend(tmp)
            else:
                text1.append(tmp)

        text1.insert(0, special_tokens["cls_token"])
        if mask_exists == -1:
            text1.append(special_tokens["mask_token"])

        text1.append(special_tokens["sep_token"])

        token = tokenize(text1, padding)
        return token


def main():
    preprocess_text()

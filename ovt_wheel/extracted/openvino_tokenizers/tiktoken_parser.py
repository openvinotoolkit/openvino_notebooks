from typing import Optional

from tiktoken import Encoding


def bpe(mergeable_ranks: dict[bytes, int], token: bytes, max_rank: Optional[int] = None) -> list[bytes]:
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        if min_idx is None:
            raise ValueError(f"Tiktoken conversion error: cannot determine bpe for token {token}.")
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2 :]
    return parts


def generate_vocab_and_merges(
    encoding: Encoding,
) -> tuple[dict[bytes, int], list[tuple[bytes, bytes]], dict[str, int]]:
    mergeable_ranks = encoding._mergeable_ranks

    vocab = {}
    merges = []
    added_tokens = {}

    for token, rank in mergeable_ranks.items():
        vocab[token] = rank

        if len(token) == 1:
            continue
        merged = tuple(bpe(mergeable_ranks, token, max_rank=rank))

        #  if special tokens added to the tokenizer and the bpe split might produce more than 2 tokens
        #  if there are "\t" in the vocab and special token "\t\t\t" was added before "\t\t" it will
        #  be tokenized into 3 tokens: bpe("\t\t\t") -> ["\t", "\t", "\t"] which is cannot be included
        #  in merges
        if len(merged) == 2:
            merges.append(merged)
        else:
            added_tokens[token] = rank

    # Also add special tokens
    vocab.update({string.encode(): idx for string, idx in encoding._special_tokens.items()})

    return vocab, merges, added_tokens

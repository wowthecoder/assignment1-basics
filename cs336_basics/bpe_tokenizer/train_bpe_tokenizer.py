import regex as re
from collections import defaultdict
import multiprocessing
from functools import partial
from .pretokenization_example import find_chunk_boundaries

def pretokenization(chunk: str, special_tokens: list[str]) -> dict[tuple[bytes], int]:
    # remove from the text corpus before the next step
    special_tokens_pattern = "|".join(re.escape(tok) for tok in special_tokens)
    paragraphs = re.split(special_tokens_pattern, chunk)

    # Pre-tokenziation
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    token_freqs: dict[tuple[bytes], int] = defaultdict(int)

    for para in paragraphs:
        for match in re.finditer(PAT, para):
            # e.g. 'low' is converted to (l, o, w)
            token = match.group()
            tup = tuple(bytes([b]) for b in token.encode("utf-8"))
            token_freqs[tup] += 1

    return token_freqs

def build_pair_index(pair_freqs, pair_to_tokens, token_freqs):
    pair_freqs.clear()
    pair_to_tokens.clear()
    for tok_bytes, freq in token_freqs.items(): # tok_bytes is a tuple
        for j in range(len(tok_bytes) - 1):
            adj_bytes = (tok_bytes[j], tok_bytes[j+1])
            pair_freqs[adj_bytes] += freq
            pair_to_tokens[adj_bytes].add(tok_bytes)

def update_pair_counts_after_merge(merge_pair: tuple[bytes, bytes], pair_freqs, pair_to_tokens, token_freqs):
    byte1, byte2 = merge_pair
    affected_tokens = pair_to_tokens[merge_pair].copy()

    for old_token in affected_tokens:
        freq = token_freqs[old_token]
        # Remove old pair counts for this token
        for j in range(len(old_token) - 1):
            pair = (old_token[j], old_token[j + 1])
            pair_freqs[pair] -= freq
            pair_to_tokens[pair].discard(old_token)
            if pair_freqs[pair] <= 0:
                del pair_freqs[pair]
                del pair_to_tokens[pair]

        # Create new token with merged pair
        new_token = []
        i = 0
        while i < len(old_token):
            if i < len(old_token) - 1 and old_token[i] == byte1 and old_token[i+1] == byte2:
                new_token.append(byte1 + byte2)
                i += 2
            else:
                new_token.append(old_token[i])
                i += 1

        new_token_tuple = tuple(new_token)

        # Add new pair counts for the new token
        for j in range(len(new_token_tuple) - 1):
            pair = (new_token_tuple[j], new_token_tuple[j+1])
            pair_freqs[pair] += freq
            pair_to_tokens[pair].add(new_token_tuple)

        # Update token frequencies
        del token_freqs[old_token]
        token_freqs[new_token_tuple] += freq


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] = [],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a Byte Pair Encoding (BPE) tokenizer.

    Args:
        input_path (str): Path to the file containing training dat  a.
        vocab_size (int): Maximum final vocabulary size (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
        special_tokens (list[str]): List of special tokens to include in the vocabulary.

    Returns:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
        merges (list[tuple[bytes, bytes]]): A list of BPE merges produced from training. <token1> was merged with<token2>
    """
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    merges: list[tuple[bytes, bytes]] = []
    merge_start_idx = 256 + len(special_tokens)
    max_num_merges = vocab_size - merge_start_idx
    if max_num_merges < 0:
        raise ValueError("vocab_size is too small")

    # Initialize vocab with special tokens
    for i, tok in enumerate(special_tokens):
        vocab[i+256] = tok.encode("utf-8")

    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    # Pretokenization parallelization (same as before)
    num_processes = 8
    chunks = []
    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode("utf-8"))
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)

    with multiprocessing.Pool(processes=num_processes) as pool:
        pretokenize_chunk = partial(pretokenization, special_tokens=special_tokens)
        chunk_results = pool.map(pretokenize_chunk, chunks)

    # Combine all frequency dictionaries into one
    token_freqs: dict[tuple[bytes], int] = defaultdict(int)
    for chunk_freq_dict in chunk_results:
        for tok_bytes, freq in chunk_freq_dict.items():
            token_freqs[tok_bytes] += freq

    # Build initial pair frequency index
    pair_freqs: dict[tuple[bytes, bytes], int] = defaultdict(int)
    # Also maintain a reverse index: which tokens contain which pairs
    pair_to_tokens: dict[tuple[bytes, bytes], set[tuple[bytes]]] = defaultdict(set)

    build_pair_index(pair_freqs, pair_to_tokens, token_freqs)

    for i in range(max_num_merges):
        if not pair_freqs:
            print(f"No more pairs to merge at iteration {i}")
            break

        # Get the most frequent byte pair (if there's a tie, get the lexicographically greatest pair)
        merge_pair = max(pair_freqs.items(), key=lambda item: (item[1], item[0]))[0]
        byte1, byte2 = merge_pair

        # Update the states
        new_index = merge_start_idx + i
        vocab[new_index] = byte1 + byte2
        merges.append(merge_pair)

        update_pair_counts_after_merge(merge_pair, pair_freqs, pair_to_tokens, token_freqs)

    return vocab, merges

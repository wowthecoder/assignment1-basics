import regex as re
from collections import defaultdict

class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""
    vocab: dict[int, bytes]     # index -> bytes
    merges: list[tuple[bytes], int] # list of merges, each merge is a tuple of two bytes and an index

def merge(token_freqs: dict[tuple[bytes], int], pair: tuple[bytes, bytes]) -> dict[tuple[bytes], int]:
    merged_freqs = defaultdict(int)
    for tok_bytes, freq in token_freqs.items(): # tok_bytes is a tuple
        new_token = []
        i = 0
        while i < len(tok_bytes):
            # Lookahead for the pair match at position i
            if i < len(tok_bytes) - 1 and tok_bytes[i] == pair[0] and tok_bytes[i+1] == pair[1]:
                # Merge the pair
                new_token.append(tok_bytes[i] + tok_bytes[i+1])
                i += 2  # Skip the next, since it's merged
            else:
                new_token.append(tok_bytes[i])
                i += 1

        new_tok_bytes = tuple(new_token)
        # Accumulate frequency in case of collision
        merged_freqs[new_tok_bytes] += freq
                
    return merged_freqs
 
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
    
    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    # Initialize vocab with special tokens
    for i, tok in enumerate(special_tokens):
        vocab[i+256] = tok.encode("utf-8")

    # remove from the text corpus before the next step
    special_tokens_pattern = "|".join(re.escape(tok) for tok in special_tokens)
    paragraphs = re.split(special_tokens_pattern, text)

    # Pre-tokenziation
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    token_freqs: dict[tuple[bytes], int] = defaultdict(int)

    for para in paragraphs:
        for match in re.finditer(PAT, para):
            # e.g. 'low' is converted to (l, o, w)
            token = match.group()
            tup = tuple(bytes([b]) for b in token.encode("utf-8"))
            token_freqs[tup] += 1

    for i in range(max_num_merges):
        # Count the number of occurrences of all adjacent byte pairs
        adj_byte_freqs: dict[tuple[bytes, bytes], int] = defaultdict(int)
        for tok_bytes, freq in token_freqs.items(): # tok_bytes is a tuple
            for j in range(len(tok_bytes) - 1):
                adj_bytes = (tok_bytes[j], tok_bytes[j+1])
                adj_byte_freqs[adj_bytes] += freq

        # Get the most frequent byte pair (if there's a tie, get the lexicographically greatest pair)
        merge_pair = max(adj_byte_freqs.items(), key=lambda item: (item[1], item[0]))[0]
        byte1, byte2 = merge_pair

        # Update the states
        new_index = merge_start_idx + i
        vocab[new_index] = byte1 + byte2
        merges.append(merge_pair)
        token_freqs = merge(token_freqs, merge_pair)

    return vocab, merges
    
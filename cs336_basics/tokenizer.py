from typing import Iterable, Iterator
import json
import regex as re

class Tokenizer:
    def __init__(self, 
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab 
        # reverse vocab for encoding
        self.encodings = {v: k for k, v in vocab.items()}
        self.merges = merges 
        self.special_tokens = special_tokens if special_tokens else []

    # Vocab file should be a json, merge file should be a txt
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'r') as vf:
            vocab_data = json.load(vf)
            if not isinstance(vocab_data, dict):
                raise ValueError("Vocab file must contain a JSON object (dictionary).")
            vocab = {int(k): v.encode() for k, v in vocab_data.items()}

            if not merges_filepath.endswith('.txt'):
                raise ValueError("Merges file must be a .txt file.")

            # Assume that the merges are saved as a text file, each pair on a separate line
            # Take a look at tests/fixtures/gpt2_merges.txt
            with open(merges_filepath, 'r') as mf:
                merges = []
                for line in mf:
                    pair = tuple(line.strip().split())
                    merges.append(pair)
        
        return cls(vocab, merges, special_tokens)
    
    def pretokenize(self, text: str) -> list[tuple[list[bytes], bool]]:
        # Pre-tokenziation
        special_tokens_pattern = f"({'|'.join(re.escape(tok) for tok in self.special_tokens)})"
        # Split text by special tokens, keeping the separators
        parts = re.split(special_tokens_pattern, text)

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pretokens = []

        for i, part in enumerate(parts):
            if not part:  # Skip empty strings
                continue
                
            # Check if this part is a special token
            is_special = part in self.special_tokens
            
            if is_special:
                # Handle special token
                sep_token_byte_list = [bytes([b]) for b in part.encode("utf-8")]
                pretokens.append((sep_token_byte_list, True))
            else:
                # Handle regular text - tokenize it
                for match in re.finditer(PAT, part):
                    token = match.group()
                    byte_list = [bytes([b]) for b in token.encode("utf-8")]
                    pretokens.append((byte_list, False))

        return pretokens

    def encode(self, text: str) -> list[int]:
        pretokens = self.pretokenize(text)
        for pretoken, isSpecialToken in pretokens:
            if isSpecialToken:
                continue
            for merge_pair in self.merges:
                for i in range(len(pretoken) - 1):
                    if merge_pair[0] == pretoken[i] and merge_pair[1] == pretoken[i+1]:
                        pretoken[i] = merge_pair[0] + merge_pair[1]
                        pretoken.pop(i+1)
                        break 

        # Convert to int IDs
        encodings = []
        for pretoken, isSpecialToken in pretokens:
            if isSpecialToken:
                specialToken = b"".join(pretoken)
                encodings.append(self.encodings.get(specialToken, -1))
                continue
            for token in pretoken:
                encodings.append(self.encodings.get(token, -1))
        return encodings

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        return []

    def decode(self, ids: list[int]) -> str:
        tokens = [self.vocab.get(id, b'\xef\xbf\xbd') for id in ids]
        return b"".join(tokens).decode("utf-8", errors="replace")
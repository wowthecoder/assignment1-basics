import tiktoken_ext.openai_public
import inspect
import hashlib

# vocab_bpe_path = "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe"
# encoder_json_path = "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json"
# vocab_key = hashlib.sha1(vocab_bpe_path.encode()).hexdigest()
# encoder_key = hashlib.sha1(encoder_json_path.encode()).hexdigest()
# print("Vocab key:", vocab_key)
# print("Encoder key:", encoder_key)

# print(dir(tiktoken_ext.openai_public))
# The encoder we want is cl100k_base, we see this as a possible function

# print(inspect.getsource(tiktoken_ext.openai_public.gpt2))

xs = [[1, 2], [3, 4], [5, 6]]
for x in xs:
    x.append(99)  # Modifies the list object in-place
print(xs)
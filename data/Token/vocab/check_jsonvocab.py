import json
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
VOCAB_PATH = BASE_DIR / "tokenizers_bpe" / "vocab.json"

with VOCAB_PATH.open("r", encoding="utf-8") as f:
    vocab = json.load(f)

# vocab.json của BPE thường là {token: id}
print("Vocab size:", len(vocab))

# đảo lại thành id -> token
id_to_token = {v: k for k, v in vocab.items()}

print("\nFirst 20 tokens by id:")
for i in range(20):
    print(i, repr(id_to_token.get(i)))

# kiểm tra special tokens
specials = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
print("\nSpecial tokens check:")
for s in specials:
    print(s, "->", vocab.get(s))

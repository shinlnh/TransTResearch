from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
VOCAB_PATH = BASE_DIR / "tokenizers_wordpiece" / "vocab.txt"

with VOCAB_PATH.open("r", encoding="utf-8") as f:
    tokens = [line.rstrip("\n") for line in f]

print("Vocab size:", len(tokens))

print("\nFirst 20 tokens by id:")
for i, tok in enumerate(tokens[:20]):
    print(i, repr(tok))

token_to_id = {tok: i for i, tok in enumerate(tokens)}
specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"]
print("\nSpecial tokens check:")
for s in specials:
    print(s, "->", token_to_id.get(s))

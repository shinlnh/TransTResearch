from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

# ===== CONFIG ===== #
BASE_DIR = Path(__file__).resolve().parents[1]
CORPUS = BASE_DIR / "data" / "enwiki_20260101_cleaned.txt"
FALLBACK_CORPUS = BASE_DIR / "data" / "corpus_cleaned.txt"
OUT_DIR = BASE_DIR / "tokenizers_bpe"
VOCAB_SIZE = 30_000
MIN_FREQUENCY = 2


def main() -> None:
    corpus_path = CORPUS if CORPUS.exists() else FALLBACK_CORPUS
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus not found: {CORPUS} (fallback: {FALLBACK_CORPUS})"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )

    tokenizer.train([str(corpus_path)], trainer=trainer)
    # Save "vocab.json + merges.txt" (BPE vocabulary)
    tokenizer.model.save(str(OUT_DIR))
    # Save full tokenizer config (easy to reload)
    tokenizer.save(str(OUT_DIR / "tokenizer.json"))
    print("Done!")
    print("Saved to:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()

"""
Vocab using BPE : 

30k token being created from pair of word that using character group frequency in the corpus 
"""
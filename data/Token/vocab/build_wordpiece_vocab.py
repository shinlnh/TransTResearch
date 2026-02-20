from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer

# ===== CONFIG ===== #
BASE_DIR = Path(__file__).resolve().parents[1]
CORPUS = BASE_DIR / "data" / "enwiki_20260101_cleaned.txt"
FALLBACK_CORPUS = BASE_DIR / "data" / "corpus_cleaned.txt"
OUT_DIR = BASE_DIR / "tokenizers_wordpiece"
VOCAB_SIZE = 30_000
MIN_FREQUENCY = 2


def main() -> None:
    corpus_path = CORPUS if CORPUS.exists() else FALLBACK_CORPUS
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus not found: {CORPUS} (fallback: {FALLBACK_CORPUS})"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )

    tokenizer.train([str(corpus_path)], trainer=trainer)
    # Save "vocab.txt" (WordPiece vocabulary)
    tokenizer.model.save(str(OUT_DIR))
    # Save full tokenizer config (easy to reload)
    tokenizer.save(str(OUT_DIR / "tokenizer.json"))
    print("Done!")
    print("Saved to:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()

"""
30k token WordPiece vocab being created from corpus that using character group highest probability of generating text
"""
import re
from pathlib import Path

# Configuration parameters
BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = BASE_DIR / "data" / "enwiki_20260101_text"
OUTPUT_FILE = BASE_DIR / "data" / "enwiki_20260101_cleaned.txt"

INPUT_PATTERNS = ("part*.txt",)
INCLUDE_WIKI_SUBDIRS = False  # Include INPUT_DIR/**/wiki_* if True.

ENCODING = "utf-8"
READ_ERRORS = "ignore"  # "ignore" or "replace"

LOWERCASE = False
KEEP_REDIRECT = False
MIN_LEN = 20

# ----------------
redirect_re = re.compile(r"^\s*#redirect\b", re.IGNORECASE)


def iter_input_files() -> list[Path]:
    files: list[Path] = []
    for pattern in INPUT_PATTERNS:
        files.extend(INPUT_DIR.glob(pattern))
    if INCLUDE_WIKI_SUBDIRS:
        files.extend(INPUT_DIR.rglob("wiki_*"))
    # De-duplicate and keep a stable order.
    unique = sorted({p.resolve() for p in files})
    return unique


def normalize_line(line: str) -> str:
    line = line.strip()
    if not line:
        return ""
    if (not KEEP_REDIRECT) and redirect_re.match(line):
        return ""
    line = re.sub(r"\s+", " ", line)

    line = "".join(ch for ch in line if ch.isprintable())

    if LOWERCASE:
        line = line.lower()

    if len(line) < MIN_LEN:
        return ""

    return line


def main() -> None:
    input_files = iter_input_files()
    if not input_files:
        raise FileNotFoundError(
            f"No input files found in {INPUT_DIR.resolve()} "
            f"(patterns: {', '.join(INPUT_PATTERNS)})"
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0

    with OUTPUT_FILE.open("w", encoding=ENCODING, newline="\n") as out:
        for fp in input_files:
            with fp.open("r", encoding=ENCODING, errors=READ_ERRORS) as f:
                for line in f:
                    clean = normalize_line(line)
                    if clean:
                        out.write(clean + "\n")
                        kept += 1
                    else:
                        skipped += 1

    print("Done.")
    print(f"Input files: {len(input_files)}")
    print(f"Kept lines: {kept:,}")
    print(f"Skipped lines: {skipped:,}")
    print(f"Output: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
